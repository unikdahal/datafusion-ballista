// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Producer-owned metadata. Only accepted successful task attempts may commit.
//! A visible block is immutable within its generation; losing one requires a
//! rollover. Empty producing snapshots mean WAIT, never end of input.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use ballista_core::JobId;
use ballista_core::error::{BallistaError, Result};
use ballista_core::serde::scheduler::PartitionLocation;
use tokio::sync::Notify;

/// Logical identity of one committed shuffle artifact.
///
/// Ballista writers may emit multiple files for the same producer task and
/// output partition. `file_id` therefore participates in artifact identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShuffleBlockKey {
    /// Scheduler task slot that produced the artifact.
    pub producer_task_id: usize,
    /// Logical output partition read by downstream stages.
    pub output_partition_id: usize,
    /// Stable file discriminator within one task/output-partition pair.
    pub file_id: Option<u64>,
}

impl From<&PartitionLocation> for ShuffleBlockKey {
    fn from(location: &PartitionLocation) -> Self {
        Self {
            producer_task_id: location.map_partition_id,
            output_partition_id: location.partition_id.partition_id,
            file_id: location.file_id,
        }
    }
}

/// Lifecycle of one generation of producer-owned shuffle input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShuffleInputLifecycle {
    /// Producers may still publish successful task results.
    Producing,
    /// The producer stage completed and no new task result may be published.
    Sealed,
}

/// A committed shuffle location annotated with the version that exposed it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishedShuffleLocation {
    /// Generation-local version at which this location became visible.
    pub published_version: u64,
    /// Immutable materialized shuffle location.
    pub location: PartitionLocation,
}

/// Point-in-time view of a generation after a consumer cursor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShuffleInputSnapshot {
    /// Generation from which the snapshot was read.
    pub generation: u64,
    /// Current generation-local version.
    pub version: u64,
    /// Current producer lifecycle.
    pub lifecycle: ShuffleInputLifecycle,
    /// Locations published after the requested cursor.
    pub locations: Vec<PublishedShuffleLocation>,
}

/// Result of reading a generation-pinned shuffle input.
///
/// A generation mismatch terminates the old consumer attempt; consumers must
/// never silently switch to the new generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShuffleInputRead {
    /// Current state and locations newer than the supplied cursor.
    Update(ShuffleInputSnapshot),
    /// The pinned generation was replaced by a newer history.
    Invalidated {
        /// Generation requested by the consumer.
        expected: u64,
        /// Current producer generation.
        current: u64,
    },
    /// The consumer cursor is ahead of the producer's current version.
    CursorAhead {
        /// Version requested by the consumer.
        after: u64,
        /// Current producer version.
        current: u64,
    },
    /// The job/input no longer exists in the registry.
    Closed,
}

/// Mutable metadata for one producer stage's current generation.
#[derive(Debug)]
pub struct ShuffleGenerationState {
    generation: u64,
    version: u64,
    lifecycle: ShuffleInputLifecycle,
    blocks: HashMap<ShuffleBlockKey, PublishedShuffleLocation>,
    // Successful task publication is the idempotence boundary, including an
    // accepted empty result. Values are canonicalized by ShuffleBlockKey.
    accepted_tasks: HashMap<usize, Vec<PartitionLocation>>,
    notify: Arc<Notify>,
}

impl ShuffleGenerationState {
    fn new(generation: u64) -> Self {
        Self {
            generation,
            version: 0,
            lifecycle: ShuffleInputLifecycle::Producing,
            blocks: HashMap::new(),
            accepted_tasks: HashMap::new(),
            notify: Arc::new(Notify::new()),
        }
    }

    /// Returns the current producer generation.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Returns whether this generation is still producing or has sealed.
    pub fn lifecycle(&self) -> ShuffleInputLifecycle {
        self.lifecycle
    }

    /// Returns true when at least one materialized shuffle artifact is visible.
    pub fn has_committed_input(&self) -> bool {
        !self.blocks.is_empty()
    }
}

/// A fully validated registry mutation. The scheduler prepares this while
/// holding the registry lock, performs the corresponding infallible graph
/// transition, then applies the token before releasing the same lock.
#[derive(Debug)]
pub(crate) struct PreparedShuffleCommit {
    job: JobId,
    stage: usize,
    generation: u64,
    base_version: u64,
    version: u64,
    task: usize,
    publication: Vec<PartitionLocation>,
    replay: bool,
}

/// Scheduler-owned registry for generation-pinned committed shuffle metadata.
///
/// Mutations are serialized by scheduler ownership. Callers must never hold the
/// registry lock while awaiting a notification. Register the notification
/// before reading state to avoid lost wakeups.
#[derive(Debug, Default)]
pub struct ShuffleInputRegistry {
    inputs: HashMap<(JobId, usize), ShuffleGenerationState>,
}

impl ShuffleInputRegistry {
    /// Initializes a stage once.
    ///
    /// Reopening a closed job must use a fresh job identity.
    pub fn create(&mut self, job: &JobId, stage: usize) {
        self.inputs
            .entry((job.clone(), stage))
            .or_insert_with(|| ShuffleGenerationState::new(1));
    }

    /// Returns the current generation state for a producer stage.
    pub fn get(&self, job: &JobId, stage: usize) -> Option<&ShuffleGenerationState> {
        self.inputs.get(&(job.clone(), stage))
    }

    /// Returns the notification handle used to wake generation readers.
    pub fn notification(&self, job: &JobId, stage: usize) -> Option<Arc<Notify>> {
        self.get(job, stage).map(|state| state.notify.clone())
    }

    /// Validate one complete successful task publication without mutating state.
    ///
    /// The registry lock must remain held between prepare and commit. Exact
    /// replay is represented explicitly and never advances the version.
    pub(crate) fn prepare_commit(
        &self,
        job: &JobId,
        stage: usize,
        generation: u64,
        task: usize,
        locations: Vec<PartitionLocation>,
    ) -> Result<PreparedShuffleCommit> {
        let publication = canonical_publication(job, stage, task, locations)?;
        let state = self.current(job, stage, generation)?;

        if let Some(previous) = state.accepted_tasks.get(&task) {
            if previous == &publication {
                return Ok(PreparedShuffleCommit {
                    job: job.clone(),
                    stage,
                    generation,
                    base_version: state.version,
                    version: state.version,
                    task,
                    publication,
                    replay: true,
                });
            }
            return Err(invariant(
                "conflicting successful task publication in one generation",
            ));
        }

        if state.lifecycle == ShuffleInputLifecycle::Sealed {
            return Err(invariant("publication after shuffle seal"));
        }

        for location in &publication {
            if state.blocks.contains_key(&ShuffleBlockKey::from(location)) {
                return Err(invariant(
                    "shuffle artifact already exists without accepted task publication",
                ));
            }
        }

        let version = if publication.is_empty() {
            state.version
        } else {
            increment(state.version)?
        };

        Ok(PreparedShuffleCommit {
            job: job.clone(),
            stage,
            generation,
            base_version: state.version,
            version,
            task,
            publication,
            replay: false,
        })
    }

    /// Apply a publication already validated by `prepare_commit`.
    ///
    /// This is intentionally infallible while the caller retains the same
    /// registry lock across prepare -> graph transition -> commit.
    pub(crate) fn commit_prepared(&mut self, prepared: PreparedShuffleCommit) -> u64 {
        let state = self
            .inputs
            .get_mut(&(prepared.job.clone(), prepared.stage))
            .expect("prepared shuffle input disappeared while registry lock was held");
        assert_eq!(
            state.generation, prepared.generation,
            "prepared shuffle generation changed while registry lock was held"
        );
        assert_eq!(
            state.version, prepared.base_version,
            "prepared shuffle version changed while registry lock was held"
        );

        if prepared.replay {
            return state.version;
        }

        if prepared.publication.is_empty() {
            state
                .accepted_tasks
                .insert(prepared.task, prepared.publication);
            return state.version;
        }

        for location in &prepared.publication {
            state.blocks.insert(
                ShuffleBlockKey::from(location),
                PublishedShuffleLocation {
                    published_version: prepared.version,
                    location: location.clone(),
                },
            );
        }
        state
            .accepted_tasks
            .insert(prepared.task, prepared.publication);
        state.version = prepared.version;
        state.notify.notify_waiters();
        state.version
    }

    /// Validate and commit in one call for callers that do not need to couple
    /// publication with another state transition.
    pub fn commit(
        &mut self,
        job: &JobId,
        stage: usize,
        generation: u64,
        task: usize,
        locations: Vec<PartitionLocation>,
    ) -> Result<u64> {
        let prepared = self.prepare_commit(job, stage, generation, task, locations)?;
        Ok(self.commit_prepared(prepared))
    }

    /// Seals the current generation at the producer's final-success transition.
    pub fn seal(&mut self, job: &JobId, stage: usize, generation: u64) -> Result<()> {
        let state = self.current_mut(job, stage, generation)?;
        if state.lifecycle != ShuffleInputLifecycle::Sealed {
            let version = increment(state.version)?;
            state.version = version;
            state.lifecycle = ShuffleInputLifecycle::Sealed;
            state.notify.notify_waiters();
        }
        Ok(())
    }

    /// Rolls to a new generation after materialized output is lost.
    ///
    /// Retry granularity is a producer task, so naming any lost artifact drops
    /// every artifact and the accepted-publication record for that producer
    /// task. Surviving tasks are retained and re-versioned as generation-local
    /// history. An unpublished task failure does not call this method.
    pub fn invalidate(
        &mut self,
        job: &JobId,
        stage: usize,
        generation: u64,
        lost: &HashSet<ShuffleBlockKey>,
    ) -> Result<u64> {
        let lost_tasks: HashSet<_> =
            lost.iter().map(|key| key.producer_task_id).collect();
        self.invalidate_tasks(job, stage, generation, &lost_tasks)
    }

    fn invalidate_tasks(
        &mut self,
        job: &JobId,
        stage: usize,
        generation: u64,
        lost_tasks: &HashSet<usize>,
    ) -> Result<u64> {
        let state = self.current_mut(job, stage, generation)?;
        if lost_tasks.is_empty() {
            return Ok(state.generation);
        }

        let next = increment(state.generation)?;
        state
            .blocks
            .retain(|key, _| !lost_tasks.contains(&key.producer_task_id));
        state
            .accepted_tasks
            .retain(|task, _| !lost_tasks.contains(task));
        for block in state.blocks.values_mut() {
            block.published_version = 1;
        }
        state.generation = next;
        state.version = 1;
        state.lifecycle = ShuffleInputLifecycle::Producing;
        state.notify.notify_waiters();
        Ok(next)
    }

    /// Reads locations newer than `after` for the requested output partitions.
    ///
    /// The selection is normalized once so scanning B committed blocks costs
    /// O(P + B), rather than O(P * B), for P requested partitions.
    pub fn read(
        &self,
        job: &JobId,
        stage: usize,
        generation: u64,
        after: u64,
        partitions: &[usize],
    ) -> Result<ShuffleInputRead> {
        let partitions: HashSet<_> = partitions.iter().copied().collect();
        self.read_selected(job, stage, generation, after, &partitions)
    }

    /// Reads using an already-normalized partition set.
    ///
    /// RPC callers can build this set before taking the scheduler registry lock
    /// to keep request-normalization work out of the critical section.
    pub fn read_selected(
        &self,
        job: &JobId,
        stage: usize,
        generation: u64,
        after: u64,
        partitions: &HashSet<usize>,
    ) -> Result<ShuffleInputRead> {
        let Some(state) = self.get(job, stage) else {
            return Ok(ShuffleInputRead::Closed);
        };
        if generation != state.generation {
            return Ok(ShuffleInputRead::Invalidated {
                expected: generation,
                current: state.generation,
            });
        }
        if after > state.version {
            return Ok(ShuffleInputRead::CursorAhead {
                after,
                current: state.version,
            });
        }

        let mut locations: Vec<_> = state
            .blocks
            .values()
            .filter(|block| {
                block.published_version > after
                    && partitions.contains(&block.location.partition_id.partition_id)
            })
            .cloned()
            .collect();
        locations.sort_by_key(|block| {
            (
                block.published_version,
                ShuffleBlockKey::from(&block.location),
            )
        });
        Ok(ShuffleInputRead::Update(ShuffleInputSnapshot {
            generation: state.generation,
            version: state.version,
            lifecycle: state.lifecycle,
            locations,
        }))
    }

    /// Removes every stage for a job and wakes current waiters before removal.
    pub fn close_job(&mut self, job: &JobId) {
        self.inputs.retain(|(id, _), state| {
            if id == job {
                state.notify.notify_waiters();
                false
            } else {
                true
            }
        });
    }

    /// Reconcile with accepted task state after existing lost-output recovery.
    pub fn retain_tasks(
        &mut self,
        job: &JobId,
        stage: usize,
        accepted: &HashSet<usize>,
    ) -> Result<bool> {
        let Some(state) = self.get(job, stage) else {
            return Ok(false);
        };
        // An accepted empty publication is still part of the sealed history.
        // Losing it must reopen that history just like losing a materialized
        // block; otherwise the replacement task cannot commit after seal.
        let lost: HashSet<_> = state
            .accepted_tasks
            .keys()
            .filter(|task| !accepted.contains(task))
            .copied()
            .collect();
        if lost.is_empty() {
            return Ok(false);
        }
        self.invalidate_tasks(job, stage, state.generation, &lost)?;
        Ok(true)
    }

    /// Compatibility snapshot for stage rollback and the blocking planner.
    pub fn stage_output(
        &self,
        job: &JobId,
        stage: usize,
    ) -> Option<super::execution_stage::StageOutput> {
        let state = self.get(job, stage)?;
        let mut output = super::execution_stage::StageOutput::new();
        for block in state.blocks.values() {
            output.add_partition(block.location.clone());
        }
        output.complete = state.lifecycle == ShuffleInputLifecycle::Sealed;
        Some(output)
    }

    fn current(
        &self,
        job: &JobId,
        stage: usize,
        generation: u64,
    ) -> Result<&ShuffleGenerationState> {
        let state = self
            .inputs
            .get(&(job.clone(), stage))
            .ok_or_else(|| invariant("shuffle input closed"))?;
        if state.generation != generation {
            return Err(invariant("stale shuffle generation publication"));
        }
        Ok(state)
    }

    fn current_mut(
        &mut self,
        job: &JobId,
        stage: usize,
        generation: u64,
    ) -> Result<&mut ShuffleGenerationState> {
        let state = self
            .inputs
            .get_mut(&(job.clone(), stage))
            .ok_or_else(|| invariant("shuffle input closed"))?;
        if state.generation != generation {
            return Err(invariant("stale shuffle generation publication"));
        }
        Ok(state)
    }
}

fn canonical_publication(
    job: &JobId,
    stage: usize,
    task: usize,
    locations: Vec<PartitionLocation>,
) -> Result<Vec<PartitionLocation>> {
    let mut by_key = HashMap::with_capacity(locations.len());
    for location in locations {
        if location.partition_id.job_id != *job
            || location.partition_id.stage_id != stage
            || location.map_partition_id != task
        {
            return Err(invariant("publication has a different producer identity"));
        }

        let key = ShuffleBlockKey::from(&location);
        if let Some(previous) = by_key.get(&key) {
            if previous != &location {
                return Err(invariant(
                    "conflicting shuffle artifact identity in one task publication",
                ));
            }
        } else {
            by_key.insert(key, location);
        }
    }

    let mut publication: Vec<_> = by_key.into_values().collect();
    publication.sort_by_key(|location| ShuffleBlockKey::from(location));
    Ok(publication)
}

fn invariant(message: &str) -> BallistaError {
    BallistaError::Internal(message.to_owned())
}

fn increment(value: u64) -> Result<u64> {
    value
        .checked_add(1)
        .ok_or_else(|| invariant("shuffle sequence exhausted"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ballista_core::serde::scheduler::{
        ExecutorMetadata, PartitionId, PartitionStats,
    };

    fn location(task: usize, partition: usize) -> PartitionLocation {
        PartitionLocation {
            map_partition_id: task,
            partition_id: PartitionId::new(&JobId::from("job"), 1, partition),
            executor_meta: ExecutorMetadata {
                id: "executor".into(),
                host: "localhost".into(),
                port: 50051,
                grpc_port: 50052,
                specification: Default::default(),
                os_info: Default::default(),
            },
            partition_stats: PartitionStats::default(),
            file_id: Some(task as u64),
            is_sort_shuffle: true,
        }
    }

    #[test]
    fn task_publication_is_atomic_idempotent_and_order_independent() -> Result<()> {
        let job = JobId::from("job");
        let mut registry = ShuffleInputRegistry::default();
        registry.create(&job, 1);

        let blocks = vec![location(0, 0), location(0, 1)];
        assert_eq!(registry.commit(&job, 1, 1, 0, blocks.clone())?, 1);
        assert_eq!(
            registry.commit(&job, 1, 1, 0, vec![blocks[1].clone(), blocks[0].clone()],)?,
            1
        );

        let mut conflict = location(0, 1);
        conflict.executor_meta.id = "different-executor".into();
        assert!(
            registry
                .commit(&job, 1, 1, 0, vec![location(0, 0), conflict])
                .is_err()
        );
        assert!(
            registry
                .commit(
                    &job,
                    1,
                    1,
                    0,
                    vec![location(0, 0), location(0, 1), location(0, 2)],
                )
                .is_err()
        );

        let ShuffleInputRead::Update(snapshot) =
            registry.read(&job, 1, 1, 0, &[0, 1, 2])?
        else {
            panic!()
        };
        assert_eq!(snapshot.version, 1);
        assert_eq!(snapshot.locations.len(), 2);
        assert!(snapshot.locations.iter().all(|b| b.published_version == 1));

        registry.commit(&job, 1, 1, 1, vec![location(1, 0)])?;
        assert_eq!(registry.commit(&job, 1, 1, 0, blocks.clone())?, 2);

        registry.seal(&job, 1, 1)?;
        assert!(
            registry
                .commit(&job, 1, 1, 2, vec![location(2, 0)])
                .is_err()
        );
        assert!(registry.commit(&job, 1, 1, 2, vec![]).is_err());
        // An already-accepted success status remains a harmless replay after seal.
        assert_eq!(registry.commit(&job, 1, 1, 0, blocks)?, 3);
        Ok(())
    }

    #[test]
    fn empty_task_publication_is_remembered() -> Result<()> {
        let job = JobId::from("job");
        let mut registry = ShuffleInputRegistry::default();
        registry.create(&job, 1);

        assert_eq!(registry.commit(&job, 1, 1, 0, vec![])?, 0);
        assert_eq!(registry.commit(&job, 1, 1, 0, vec![])?, 0);
        assert!(
            registry
                .commit(&job, 1, 1, 0, vec![location(0, 0)])
                .is_err()
        );

        assert_eq!(registry.commit(&job, 1, 1, 1, vec![location(1, 0)])?, 1);
        // Replay returns the current global version but does not advance it.
        assert_eq!(registry.commit(&job, 1, 1, 0, vec![])?, 1);
        Ok(())
    }

    #[test]
    fn multiple_files_for_one_task_partition_are_distinct_artifacts() -> Result<()> {
        let job = JobId::from("job");
        let mut registry = ShuffleInputRegistry::default();
        registry.create(&job, 1);

        let mut first = location(0, 0);
        first.file_id = Some(10);
        let mut second = first.clone();
        second.file_id = Some(11);
        assert_ne!(
            ShuffleBlockKey::from(&first),
            ShuffleBlockKey::from(&second)
        );

        assert_eq!(registry.commit(&job, 1, 1, 0, vec![first, second])?, 1);
        let ShuffleInputRead::Update(snapshot) = registry.read(&job, 1, 1, 0, &[0])?
        else {
            panic!()
        };
        assert_eq!(snapshot.locations.len(), 2);
        Ok(())
    }

    #[test]
    fn rollover_is_task_granular_and_preserves_other_tasks() -> Result<()> {
        let job = JobId::from("job");
        let mut registry = ShuffleInputRegistry::default();
        registry.create(&job, 1);

        let task_zero = vec![location(0, 0), location(0, 1)];
        let task_one = vec![location(1, 0)];
        registry.commit(&job, 1, 1, 0, task_zero.clone())?;
        registry.commit(&job, 1, 1, 1, task_one.clone())?;

        // Losing one artifact invalidates the whole producer task because retry
        // happens at task granularity.
        let lost = HashSet::from([ShuffleBlockKey::from(&task_zero[0])]);
        assert_eq!(registry.invalidate(&job, 1, 1, &lost)?, 2);

        assert!(matches!(
            registry.read(&job, 1, 1, 0, &[0])?,
            ShuffleInputRead::Invalidated {
                expected: 1,
                current: 2
            }
        ));
        assert!(registry.commit(&job, 1, 1, 0, task_zero.clone()).is_err());
        assert!(registry.seal(&job, 1, 1).is_err());

        let ShuffleInputRead::Update(snapshot) = registry.read(&job, 1, 2, 0, &[0, 1])?
        else {
            panic!()
        };
        assert_eq!(snapshot.generation, 2);
        assert_eq!(snapshot.version, 1);
        assert_eq!(snapshot.locations.len(), 1);
        assert_eq!(snapshot.locations[0].location.map_partition_id, 1);
        assert_eq!(snapshot.locations[0].published_version, 1);

        // The retained task is still an accepted exact replay, while the lost
        // task may publish a fresh complete result in the new generation.
        assert_eq!(registry.commit(&job, 1, 2, 1, task_one)?, 1);
        assert_eq!(registry.commit(&job, 1, 2, 0, task_zero)?, 2);
        Ok(())
    }

    #[test]
    fn rollover_can_drop_all_materialized_blocks() -> Result<()> {
        let job = JobId::from("job");
        let mut registry = ShuffleInputRegistry::default();
        registry.create(&job, 1);

        let block = location(0, 0);
        registry.commit(&job, 1, 1, 0, vec![block.clone()])?;
        let lost = HashSet::from([ShuffleBlockKey::from(&block)]);
        assert_eq!(registry.invalidate(&job, 1, 1, &lost)?, 2);

        let ShuffleInputRead::Update(snapshot) = registry.read(&job, 1, 2, 0, &[0])?
        else {
            panic!()
        };
        assert!(snapshot.locations.is_empty());
        assert_eq!(snapshot.version, 1);
        assert_eq!(snapshot.lifecycle, ShuffleInputLifecycle::Producing);
        assert!(!registry.get(&job, 1).unwrap().has_committed_input());
        Ok(())
    }

    #[test]
    fn overflow_fails_closed_without_partial_mutation() -> Result<()> {
        let job = JobId::from("job");
        let mut registry = ShuffleInputRegistry::default();
        registry.create(&job, 1);

        {
            let state = registry.inputs.get_mut(&(job.clone(), 1)).unwrap();
            state.version = u64::MAX;
        }
        assert!(
            registry
                .commit(&job, 1, 1, 0, vec![location(0, 0)])
                .is_err()
        );
        let state = registry.get(&job, 1).unwrap();
        assert!(state.blocks.is_empty());
        assert!(state.accepted_tasks.is_empty());
        assert_eq!(state.lifecycle, ShuffleInputLifecycle::Producing);
        assert!(registry.seal(&job, 1, 1).is_err());
        assert_eq!(
            registry.get(&job, 1).unwrap().lifecycle,
            ShuffleInputLifecycle::Producing
        );

        let mut registry = ShuffleInputRegistry::default();
        registry.create(&job, 1);
        let block = location(0, 0);
        registry.commit(&job, 1, 1, 0, vec![block.clone()])?;
        {
            let state = registry.inputs.get_mut(&(job.clone(), 1)).unwrap();
            state.generation = u64::MAX;
        }
        let lost = HashSet::from([ShuffleBlockKey::from(&block)]);
        assert!(registry.invalidate(&job, 1, u64::MAX, &lost).is_err());
        let state = registry.get(&job, 1).unwrap();
        assert_eq!(state.generation, u64::MAX);
        assert_eq!(state.blocks.len(), 1);
        assert_eq!(state.accepted_tasks.len(), 1);
        Ok(())
    }

    #[test]
    fn retain_tasks_drops_entire_retired_attempt_and_replays_survivors() -> Result<()> {
        let job = JobId::from("job");
        let mut registry = ShuffleInputRegistry::default();
        registry.create(&job, 1);

        registry.commit(&job, 1, 1, 0, vec![location(0, 0), location(0, 1)])?;
        registry.commit(&job, 1, 1, 1, vec![location(1, 0), location(1, 1)])?;

        assert!(registry.retain_tasks(&job, 1, &HashSet::from([1]))?);
        assert_eq!(registry.get(&job, 1).unwrap().generation(), 2);
        let ShuffleInputRead::Update(snapshot) = registry.read(&job, 1, 2, 0, &[0, 1])?
        else {
            panic!()
        };
        assert_eq!(snapshot.locations.len(), 2);
        assert!(
            snapshot
                .locations
                .iter()
                .all(|block| block.location.map_partition_id == 1)
        );

        // A successful retry gets a fresh append-only task ID. Reconciliation
        // must retain both the surviving old task and the accepted retry, never
        // resurrect task 0 from the retired attempt.
        registry.commit(&job, 1, 2, 2, vec![location(2, 0), location(2, 1)])?;
        assert!(!registry.retain_tasks(&job, 1, &HashSet::from([1, 2]))?);
        let ShuffleInputRead::Update(snapshot) = registry.read(&job, 1, 2, 0, &[0, 1])?
        else {
            panic!()
        };
        let producers: HashSet<_> = snapshot
            .locations
            .iter()
            .map(|block| block.location.map_partition_id)
            .collect();
        assert_eq!(producers, HashSet::from([1, 2]));

        assert!(registry.retain_tasks(&job, 1, &HashSet::from([2]))?);
        assert_eq!(registry.get(&job, 1).unwrap().generation(), 3);
        let ShuffleInputRead::Update(snapshot) = registry.read(&job, 1, 3, 0, &[0, 1])?
        else {
            panic!()
        };
        assert_eq!(snapshot.locations.len(), 2);
        assert!(
            snapshot
                .locations
                .iter()
                .all(|block| block.location.map_partition_id == 2)
        );
        Ok(())
    }

    #[test]
    fn empty_producing_input_is_not_sealed() -> Result<()> {
        let job = JobId::from("job");
        let mut registry = ShuffleInputRegistry::default();
        registry.create(&job, 1);
        let ShuffleInputRead::Update(snapshot) = registry.read(&job, 1, 1, 0, &[0])?
        else {
            panic!()
        };
        assert!(snapshot.locations.is_empty());
        assert_eq!(snapshot.lifecycle, ShuffleInputLifecycle::Producing);
        assert!(matches!(
            registry.read(&job, 1, 1, 10, &[0])?,
            ShuffleInputRead::CursorAhead {
                after: 10,
                current: 0
            }
        ));

        registry.close_job(&job);
        assert!(matches!(
            registry.read(&job, 1, 1, 0, &[0])?,
            ShuffleInputRead::Closed
        ));
        Ok(())
    }

    #[test]
    fn losing_empty_publication_reopens_sealed_history() -> Result<()> {
        let job = JobId::from("job");
        let mut registry = ShuffleInputRegistry::default();
        registry.create(&job, 1);
        registry.commit(&job, 1, 1, 0, vec![])?;
        registry.commit(&job, 1, 1, 1, vec![location(1, 0)])?;
        registry.seal(&job, 1, 1)?;

        assert!(registry.retain_tasks(&job, 1, &HashSet::from([1]))?);
        assert!(matches!(
            registry.read(&job, 1, 1, 0, &[0])?,
            ShuffleInputRead::Invalidated {
                expected: 1,
                current: 2
            }
        ));
        let state = registry.get(&job, 1).unwrap();
        assert_eq!(state.lifecycle, ShuffleInputLifecycle::Producing);
        assert_eq!(state.blocks.len(), 1);
        assert!(!state.accepted_tasks.contains_key(&0));

        registry.commit(&job, 1, 2, 2, vec![])?;
        registry.seal(&job, 1, 2)?;
        assert!(!registry.retain_tasks(&job, 1, &HashSet::from([1, 2]))?);
        Ok(())
    }
}
