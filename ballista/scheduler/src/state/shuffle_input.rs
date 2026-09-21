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

/// Logical identity of a committed artifact (not an original input partition).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShuffleBlockKey {
    pub producer_task_id: usize,
    pub output_partition_id: usize,
}

impl From<&PartitionLocation> for ShuffleBlockKey {
    fn from(location: &PartitionLocation) -> Self {
        Self {
            producer_task_id: location.map_partition_id,
            output_partition_id: location.partition_id.partition_id,
        }
    }
}

/// Only a sealed and drained input permits EOF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShuffleInputLifecycle {
    Producing,
    Sealed,
}

#[derive(Debug, Clone)]
pub struct PublishedShuffleLocation {
    pub published_version: u64,
    pub location: PartitionLocation,
}

#[derive(Debug, Clone)]
pub struct ShuffleInputSnapshot {
    pub generation: u64,
    pub version: u64,
    pub lifecycle: ShuffleInputLifecycle,
    pub locations: Vec<PublishedShuffleLocation>,
}

/// A mismatch terminates the old consumer attempt; it must never switch inputs.
#[derive(Debug, Clone)]
pub enum ShuffleInputRead {
    Update(ShuffleInputSnapshot),
    Invalidated { expected: u64, current: u64 },
    Closed,
}

#[derive(Debug)]
pub struct ShuffleGenerationState {
    generation: u64,
    version: u64,
    lifecycle: ShuffleInputLifecycle,
    blocks: HashMap<ShuffleBlockKey, PublishedShuffleLocation>,
    notify: Arc<Notify>,
}

impl ShuffleGenerationState {
    fn new(generation: u64) -> Self {
        Self {
            generation,
            version: 0,
            lifecycle: ShuffleInputLifecycle::Producing,
            blocks: HashMap::new(),
            notify: Arc::new(Notify::new()),
        }
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn lifecycle(&self) -> ShuffleInputLifecycle {
        self.lifecycle
    }

    pub fn has_committed_input(&self) -> bool {
        !self.blocks.is_empty()
    }
}

/// Mutations are serialized by scheduler ownership. Never hold its lock while
/// awaiting a notification. Register the notification *before* reading state.
#[derive(Debug, Default)]
pub struct ShuffleInputRegistry {
    inputs: HashMap<(JobId, usize), ShuffleGenerationState>,
}

impl ShuffleInputRegistry {
    /// Initialize once. Reopening a closed job must use a fresh job identity.
    pub fn create(&mut self, job: &JobId, stage: usize) {
        self.inputs
            .entry((job.clone(), stage))
            .or_insert_with(|| ShuffleGenerationState::new(1));
    }

    pub fn get(&self, job: &JobId, stage: usize) -> Option<&ShuffleGenerationState> {
        self.inputs.get(&(job.clone(), stage))
    }

    pub fn notification(&self, job: &JobId, stage: usize) -> Option<Arc<Notify>> {
        self.get(job, stage).map(|state| state.notify.clone())
    }

    /// Validate the entire commit before changing any state. Exact replay does
    /// not advance the version, including an accepted empty task result.
    pub fn commit(
        &mut self,
        job: &JobId,
        stage: usize,
        generation: u64,
        task: usize,
        locations: Vec<PartitionLocation>,
    ) -> Result<u64> {
        let state = self.current_mut(job, stage, generation)?;
        let mut additions = HashMap::new();
        for location in locations {
            if location.partition_id.job_id != *job
                || location.partition_id.stage_id != stage
                || location.map_partition_id != task
            {
                return Err(invariant("publication has a different producer identity"));
            }
            let key = ShuffleBlockKey::from(&location);
            if let Some(previous) = state
                .blocks
                .get(&key)
                .map(|b| &b.location)
                .or_else(|| additions.get(&key))
            {
                if previous != &location {
                    return Err(invariant("conflicting shuffle block in one generation"));
                }
            } else {
                additions.insert(key, location);
            }
        }
        if additions.is_empty() {
            return Ok(state.version);
        }
        if state.lifecycle == ShuffleInputLifecycle::Sealed {
            return Err(invariant("publication after shuffle seal"));
        }
        let version = increment(state.version)?;
        state
            .blocks
            .extend(additions.into_iter().map(|(key, location)| {
                (
                    key,
                    PublishedShuffleLocation {
                        published_version: version,
                        location,
                    },
                )
            }));
        state.version = version;
        state.notify.notify_waiters();
        Ok(version)
    }

    /// Called only at the existing final-success stage transition.
    pub fn seal(&mut self, job: &JobId, stage: usize, generation: u64) -> Result<()> {
        let state = self.current_mut(job, stage, generation)?;
        if state.lifecycle != ShuffleInputLifecycle::Sealed {
            state.version = increment(state.version)?;
            state.lifecycle = ShuffleInputLifecycle::Sealed;
            state.notify.notify_waiters();
        }
        Ok(())
    }

    /// Retain surviving materialized blocks, but establish a new history.
    /// An unpublished task failure does not call this method.
    pub fn invalidate(
        &mut self,
        job: &JobId,
        stage: usize,
        generation: u64,
        lost: &HashSet<ShuffleBlockKey>,
    ) -> Result<u64> {
        let state = self.current_mut(job, stage, generation)?;
        let next = increment(state.generation)?;
        state.blocks.retain(|key, _| !lost.contains(key));
        for block in state.blocks.values_mut() {
            block.published_version = 1;
        }
        state.generation = next;
        state.version = 1;
        state.lifecycle = ShuffleInputLifecycle::Producing;
        state.notify.notify_waiters();
        Ok(next)
    }

    pub fn read(
        &self,
        job: &JobId,
        stage: usize,
        generation: u64,
        after: u64,
        partitions: &[usize],
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
            return Err(invariant("shuffle cursor is ahead of the producer"));
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
        locations.sort_by_key(|b| {
            (
                b.published_version,
                b.location.map_partition_id,
                b.location.partition_id.partition_id,
            )
        });
        Ok(ShuffleInputRead::Update(ShuffleInputSnapshot {
            generation: state.generation,
            version: state.version,
            lifecycle: state.lifecycle,
            locations,
        }))
    }

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
            partition_id: PartitionId::new(&"job".to_owned(), 1, partition),
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
    fn atomic_commit_replay_conflict_and_seal() -> Result<()> {
        let job = "job".to_owned();
        let mut registry = ShuffleInputRegistry::default();
        registry.create(&job, 1);
        let blocks = vec![location(0, 0), location(0, 1)];
        assert_eq!(registry.commit(&job, 1, 1, 0, blocks.clone())?, 1);
        assert_eq!(registry.commit(&job, 1, 1, 0, blocks.clone())?, 1);
        let mut conflict = location(0, 1);
        conflict.file_id = Some(99);
        assert!(
            registry
                .commit(&job, 1, 1, 0, vec![location(0, 2), conflict])
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
        registry.seal(&job, 1, 1)?;
        assert!(
            registry
                .commit(&job, 1, 1, 1, vec![location(1, 0)])
                .is_err()
        );
        // A status replay after seal remains harmless.
        assert_eq!(registry.commit(&job, 1, 1, 0, blocks)?, 2);
        let ShuffleInputRead::Update(snapshot) = registry.read(&job, 1, 1, 1, &[0])?
        else {
            panic!()
        };
        assert_eq!(snapshot.lifecycle, ShuffleInputLifecycle::Sealed);
        assert!(snapshot.locations.is_empty());
        Ok(())
    }

    #[test]
    fn rollover_preserves_survivors_and_rejects_stale_attempts() -> Result<()> {
        let job = "job".to_owned();
        let mut registry = ShuffleInputRegistry::default();
        registry.create(&job, 1);
        registry.commit(&job, 1, 1, 0, vec![location(0, 0), location(0, 1)])?;
        let lost = HashSet::from([ShuffleBlockKey::from(&location(0, 0))]);
        assert_eq!(registry.invalidate(&job, 1, 1, &lost)?, 2);
        assert!(matches!(
            registry.read(&job, 1, 1, 0, &[0])?,
            ShuffleInputRead::Invalidated {
                expected: 1,
                current: 2
            }
        ));
        assert!(
            registry
                .commit(&job, 1, 1, 0, vec![location(0, 0)])
                .is_err()
        );
        let ShuffleInputRead::Update(snapshot) = registry.read(&job, 1, 2, 0, &[0, 1])?
        else {
            panic!()
        };
        assert_eq!(snapshot.locations.len(), 1);
        assert_eq!(snapshot.locations[0].location.partition_id.partition_id, 1);
        registry.close_job(&job);
        assert!(matches!(
            registry.read(&job, 1, 2, 0, &[0])?,
            ShuffleInputRead::Closed
        ));
        Ok(())
    }

    #[test]
    fn empty_producing_input_is_not_sealed() -> Result<()> {
        let job = "job".to_owned();
        let mut registry = ShuffleInputRegistry::default();
        registry.create(&job, 1);
        let ShuffleInputRead::Update(snapshot) = registry.read(&job, 1, 1, 0, &[0])?
        else {
            panic!()
        };
        assert!(snapshot.locations.is_empty());
        assert_eq!(snapshot.lifecycle, ShuffleInputLifecycle::Producing);
        assert!(registry.read(&job, 1, 1, 10, &[0]).is_err());
        Ok(())
    }
}
