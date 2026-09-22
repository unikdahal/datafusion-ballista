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

//! Canonical scheduler state for one materialized shuffle exchange.
//!
//! The state in this module is deliberately independent from ExecutionStage
//! and StageOutput. Later integration should derive those compatibility views
//! from this state rather than maintain another independently mutable copy.

use std::collections::{BTreeSet, HashMap};
use std::error::Error;
use std::fmt::{Display, Formatter};

use ballista_core::serde::scheduler::PartitionLocation;
use ballista_core::shuffle::{
    ShuffleExchangeCursor, ShuffleExchangeEpoch, ShuffleExchangeId,
    ShuffleExchangeLifecycle, ShuffleExchangeSequence,
};

/// Logical identity of one immutable shuffle artifact within an exchange epoch.
///
/// file_id participates in identity because one producer task may emit more
/// than one physical artifact for the same output partition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ShuffleArtifactKey {
    producer_task_id: usize,
    output_partition_id: usize,
    file_id: Option<u64>,
}

impl ShuffleArtifactKey {
    /// Creates the canonical key for one materialized shuffle artifact.
    pub(crate) fn from_location(location: &PartitionLocation) -> Self {
        Self {
            producer_task_id: location.map_partition_id,
            output_partition_id: location.partition_id.partition_id,
            file_id: location.file_id,
        }
    }

    /// Returns the producer task that owns this artifact.
    pub(crate) fn producer_task_id(self) -> usize {
        self.producer_task_id
    }
}

fn same_artifact_location(left: &PartitionLocation, right: &PartitionLocation) -> bool {
    left.map_partition_id == right.map_partition_id
        && left.partition_id == right.partition_id
        && left.executor_meta.id == right.executor_meta.id
        && left.executor_meta.host == right.executor_meta.host
        && left.executor_meta.port == right.executor_meta.port
        && left.executor_meta.grpc_port == right.executor_meta.grpc_port
        && left.partition_stats == right.partition_stats
        && left.file_id == right.file_id
        && left.is_sort_shuffle == right.is_sort_shuffle
}

/// One committed artifact together with the exchange event that exposed it.
#[derive(Clone, Debug)]
pub(crate) struct PublishedShuffleArtifact {
    cursor: ShuffleExchangeCursor,
    location: PartitionLocation,
}

impl PublishedShuffleArtifact {
    /// Returns the epoch-bound reader position that exposed this artifact.
    pub(crate) fn cursor(&self) -> ShuffleExchangeCursor {
        self.cursor
    }

    /// Returns the immutable artifact location.
    pub(crate) fn location(&self) -> &PartitionLocation {
        &self.location
    }
}

/// Result of accepting a producer task publication.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ShufflePublicationResult {
    /// The task publication was accepted for the first time.
    ///
    /// Empty task output is still recorded as accepted, but has no reader-visible
    /// event and therefore no event cursor.
    Committed {
        event_cursor: Option<ShuffleExchangeCursor>,
    },
    /// The scheduler observed an exact replay of a previously accepted task.
    Replay,
}

/// Description of one successful exchange-epoch rollover.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ShuffleEpochRollover {
    /// Epoch that was invalidated.
    pub(crate) previous_epoch: ShuffleExchangeEpoch,
    /// Fresh epoch containing only surviving producer output.
    pub(crate) current_epoch: ShuffleExchangeEpoch,
    /// Reader-visible bootstrap event containing every surviving artifact.
    ///
    /// This is None when no materialized artifacts survive. Accepted empty task
    /// publications can still survive without creating a reader-visible event.
    pub(crate) bootstrap_cursor: Option<ShuffleExchangeCursor>,
    /// Accepted producer tasks removed by this rollover, in deterministic order.
    pub(crate) invalidated_tasks: Vec<usize>,
    /// Number of accepted producer tasks retained in the new epoch.
    pub(crate) retained_task_count: usize,
    /// Number of materialized artifacts retained in the new epoch.
    pub(crate) retained_artifact_count: usize,
}

/// Result of applying an output-loss signal to an exchange.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ShuffleInvalidationResult {
    /// No currently accepted producer output matched the loss signal.
    Unchanged,
    /// At least one accepted producer task was invalidated and the epoch rolled.
    Rolled(ShuffleEpochRollover),
}

/// Structural exchange-state error.
///
/// These variants are intentionally specific so later scheduler integration can
/// decide whether an error is a stale report, a retryable recovery condition, or
/// an internal invariant violation without parsing strings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ShuffleExchangeError {
    /// An operation targeted an epoch that is no longer current.
    StaleEpoch {
        requested: ShuffleExchangeEpoch,
        current: ShuffleExchangeEpoch,
    },
    /// A reported artifact belongs to another logical exchange.
    WrongExchange {
        expected: ShuffleExchangeId,
        actual: ShuffleExchangeId,
    },
    /// A reported artifact names a different producer task.
    WrongProducerTask {
        expected: usize,
        actual: usize,
    },
    /// Two artifacts used the same logical key but disagreed on their contents.
    ConflictingArtifact { key: ShuffleArtifactKey },
    /// A task replay did not exactly match the task publication already accepted.
    ConflictingTaskReplay { producer_task_id: usize },
    /// A previously unseen task attempted to publish after the epoch sealed.
    PublicationAfterSeal,
    /// A reader cursor is ahead of the exchange's current event sequence.
    CursorAhead {
        after: ShuffleExchangeCursor,
        current: ShuffleExchangeCursor,
    },
    /// The exchange epoch cannot advance without wrapping.
    EpochExhausted,
    /// The event sequence cannot advance without wrapping.
    SequenceExhausted,
}

impl Display for ShuffleExchangeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StaleEpoch { requested, current } => write!(
                f,
                "stale shuffle exchange epoch: requested {}, current {}",
                requested.get(),
                current.get()
            ),
            Self::WrongExchange { expected, actual } => write!(
                f,
                "shuffle artifact belongs to {}/{}, expected {}/{}",
                actual.job_id(),
                actual.producer_stage_id(),
                expected.job_id(),
                expected.producer_stage_id()
            ),
            Self::WrongProducerTask { expected, actual } => write!(
                f,
                "shuffle artifact belongs to producer task {actual}, expected {expected}"
            ),
            Self::ConflictingArtifact { key } => {
                write!(f, "conflicting shuffle artifact for key {key:?}")
            }
            Self::ConflictingTaskReplay { producer_task_id } => write!(
                f,
                "conflicting replay for producer task {producer_task_id}"
            ),
            Self::PublicationAfterSeal => {
                f.write_str("shuffle publication attempted after exchange seal")
            }
            Self::CursorAhead { after, current } => write!(
                f,
                "shuffle cursor {}/{} is ahead of current cursor {}/{}",
                after.epoch().get(),
                after.sequence().get(),
                current.epoch().get(),
                current.sequence().get()
            ),
            Self::EpochExhausted => f.write_str("shuffle exchange epoch exhausted"),
            Self::SequenceExhausted => f.write_str("shuffle exchange sequence exhausted"),
        }
    }
}

impl Error for ShuffleExchangeError {}

type Result<T> = std::result::Result<T, ShuffleExchangeError>;

/// Canonical mutable state for one logical shuffle exchange.
///
/// All accepted producer output lives here. A producer task publication is the
/// atomic mutation boundary: either every artifact in the report becomes visible
/// at one event sequence, or the exchange remains unchanged.
///
/// Epoch rollover preserves usable producer output while invalidating every
/// reader pinned to the retired materialized history.
#[derive(Debug)]
pub(crate) struct ShuffleExchangeState {
    id: ShuffleExchangeId,
    cursor: ShuffleExchangeCursor,
    lifecycle: ShuffleExchangeLifecycle,
    artifacts: HashMap<ShuffleArtifactKey, PublishedShuffleArtifact>,
    // Output partition -> append-ordered (epoch-bound cursor, artifact key).
    partition_index:
        HashMap<usize, Vec<(ShuffleExchangeCursor, ShuffleArtifactKey)>>,
    // Producer task -> canonical artifact keys. Empty output is represented by
    // an empty vector, which is still an accepted task publication.
    //
    // Producer task IDs identify scheduler task attempts. Accepted publications
    // for surviving tasks are retained across epoch rollover; invalidated task
    // publications are removed. Tasks that had not materialized output in the
    // retired epoch may still commit into the current epoch after scheduler-side
    // task validation succeeds.
    accepted_tasks: HashMap<usize, Vec<ShuffleArtifactKey>>,
}

impl ShuffleExchangeState {
    /// Creates the initial epoch for one logical shuffle exchange.
    pub(crate) fn new(id: ShuffleExchangeId) -> Self {
        Self {
            id,
            cursor: ShuffleExchangeCursor::initial(ShuffleExchangeEpoch::INITIAL),
            lifecycle: ShuffleExchangeLifecycle::Open,
            artifacts: HashMap::new(),
            partition_index: HashMap::new(),
            accepted_tasks: HashMap::new(),
        }
    }

    /// Returns the logical exchange identity.
    pub(crate) fn id(&self) -> &ShuffleExchangeId {
        &self.id
    }

    /// Returns the current materialized-history epoch.
    pub(crate) fn epoch(&self) -> ShuffleExchangeEpoch {
        self.cursor.epoch()
    }

    /// Returns the current epoch-bound reader position.
    pub(crate) fn cursor(&self) -> ShuffleExchangeCursor {
        self.cursor
    }

    /// Returns the latest reader-visible event sequence.
    ///
    /// This is primarily useful for diagnostics and tests. Reader APIs use
    /// [ShuffleExchangeCursor] so an event position cannot be detached from
    /// the epoch whose history it indexes.
    pub(crate) fn sequence(&self) -> ShuffleExchangeSequence {
        self.cursor.sequence()
    }

    /// Returns the publication lifecycle of the current epoch.
    pub(crate) fn lifecycle(&self) -> ShuffleExchangeLifecycle {
        self.lifecycle
    }

    /// Returns whether at least one materialized artifact has been committed.
    pub(crate) fn has_committed_artifacts(&self) -> bool {
        !self.artifacts.is_empty()
    }

    /// Returns whether a producer task publication, including empty output, was accepted.
    pub(crate) fn is_task_accepted(&self, producer_task_id: usize) -> bool {
        self.accepted_tasks.contains_key(&producer_task_id)
    }

    /// Atomically accepts one complete producer-task publication.
    ///
    /// Exact replay is idempotent, including after the exchange is sealed.
    /// Conflicting replay and all malformed publications fail before any state
    /// changes. All non-empty artifacts in one accepted publication receive the
    /// same event cursor.
    ///
    /// The caller must first validate that the producer task report is still
    /// authorized by scheduler task state. It then publishes into the exchange
    /// epoch that is current at commit time. A task that had not materialized
    /// output in a retired epoch may therefore commit into the new epoch.
    pub(crate) fn publish(
        &mut self,
        epoch: ShuffleExchangeEpoch,
        producer_task_id: usize,
        locations: Vec<PartitionLocation>,
    ) -> Result<ShufflePublicationResult> {
        self.ensure_epoch(epoch)?;
        let publication = self.canonical_publication(producer_task_id, locations)?;

        if let Some(previous_keys) = self.accepted_tasks.get(&producer_task_id) {
            let exact_replay = previous_keys.len() == publication.len()
                && previous_keys.iter().zip(publication.iter()).all(
                    |(previous_key, location)| {
                        let key = ShuffleArtifactKey::from_location(location);
                        previous_key == &key
                            && self
                                .artifacts
                                .get(previous_key)
                                .is_some_and(|artifact| {
                                    same_artifact_location(&artifact.location, location)
                                })
                    },
                );

            if exact_replay {
                return Ok(ShufflePublicationResult::Replay);
            }
            return Err(ShuffleExchangeError::ConflictingTaskReplay {
                producer_task_id,
            });
        }

        if self.lifecycle.is_sealed() {
            return Err(ShuffleExchangeError::PublicationAfterSeal);
        }

        for location in &publication {
            let key = ShuffleArtifactKey::from_location(location);
            if self.artifacts.contains_key(&key) {
                return Err(ShuffleExchangeError::ConflictingArtifact { key });
            }
        }

        if publication.is_empty() {
            self.accepted_tasks.insert(producer_task_id, Vec::new());
            return Ok(ShufflePublicationResult::Committed {
                event_cursor: None,
            });
        }

        let next_cursor = self
            .cursor
            .checked_next()
            .ok_or(ShuffleExchangeError::SequenceExhausted)?;
        let mut keys = Vec::with_capacity(publication.len());

        for location in publication {
            let key = ShuffleArtifactKey::from_location(&location);
            self.artifacts.insert(
                key,
                PublishedShuffleArtifact {
                    cursor: next_cursor,
                    location,
                },
            );
            self.partition_index
                .entry(key.output_partition_id)
                .or_default()
                .push((next_cursor, key));
            keys.push(key);
        }

        self.accepted_tasks.insert(producer_task_id, keys);
        self.cursor = next_cursor;
        Ok(ShufflePublicationResult::Committed {
            event_cursor: Some(next_cursor),
        })
    }

    /// Invalidates accepted output for the specified producer tasks.
    ///
    /// Recovery is task-granular: if one artifact from a producer task is lost,
    /// every artifact and the accepted-publication record for that task must be
    /// replaced together. Unknown task IDs are ignored so duplicate loss
    /// notifications cannot churn the epoch.
    pub(crate) fn invalidate_tasks(
        &mut self,
        epoch: ShuffleExchangeEpoch,
        lost_tasks: &[usize],
    ) -> Result<ShuffleInvalidationResult> {
        self.ensure_epoch(epoch)?;

        let mut invalidated = BTreeSet::new();
        for &task in lost_tasks {
            if self.accepted_tasks.contains_key(&task) {
                invalidated.insert(task);
            }
        }
        self.roll_epoch(invalidated, false)
    }

    /// Invalidates producer tasks owning the specified materialized artifacts.
    ///
    /// Unknown artifact keys are ignored. A known artifact invalidates its whole
    /// producer task because Ballista retries producer output at task granularity.
    pub(crate) fn invalidate_artifacts(
        &mut self,
        epoch: ShuffleExchangeEpoch,
        lost_artifacts: &[ShuffleArtifactKey],
    ) -> Result<ShuffleInvalidationResult> {
        self.ensure_epoch(epoch)?;

        let mut invalidated = BTreeSet::new();
        for key in lost_artifacts {
            if self.artifacts.contains_key(key) {
                invalidated.insert(key.producer_task_id());
            }
        }
        self.roll_epoch(invalidated, false)
    }

    /// Invalidates every accepted publication in the current epoch.
    ///
    /// This is the stage-attempt rollback primitive. Once the computation that
    /// produced an exchange is discarded, no publication from that attempt may
    /// survive merely because scheduler task status was already rewritten to
    /// ResultLost/TaskKilled. Accepted empty publications are included.
    ///
    /// Unlike task/artifact loss, this always rolls the epoch even when no
    /// publication has been accepted yet. A full producer-attempt reset retires
    /// the generation itself: future incremental consumers may already be
    /// subscribed to an open empty exchange and must be able to observe that
    /// generation change.
    pub(crate) fn invalidate_all(
        &mut self,
        epoch: ShuffleExchangeEpoch,
    ) -> Result<ShuffleInvalidationResult> {
        self.ensure_epoch(epoch)?;
        let invalidated = self.accepted_tasks.keys().copied().collect();
        self.roll_epoch(invalidated, true)
    }

    /// Seals the current epoch.
    ///
    /// The first seal is reader-visible and consumes one event sequence.
    /// Repeated seal calls are idempotent and return None.
    pub(crate) fn seal(
        &mut self,
        epoch: ShuffleExchangeEpoch,
    ) -> Result<Option<ShuffleExchangeCursor>> {
        self.ensure_epoch(epoch)?;
        if self.lifecycle.is_sealed() {
            return Ok(None);
        }

        let next_cursor = self
            .cursor
            .checked_next()
            .ok_or(ShuffleExchangeError::SequenceExhausted)?;
        self.lifecycle = ShuffleExchangeLifecycle::Sealed;
        self.cursor = next_cursor;
        Ok(Some(next_cursor))
    }

    /// Returns artifacts for one output partition that became visible after after.
    ///
    /// The returned references are ordered by publication sequence and then by
    /// canonical artifact key within a task-atomic publication.
    pub(crate) fn artifacts_after(
        &self,
        after: ShuffleExchangeCursor,
        output_partition_id: usize,
    ) -> Result<Vec<&PublishedShuffleArtifact>> {
        self.ensure_epoch(after.epoch())?;
        if after.sequence() > self.cursor.sequence() {
            return Err(ShuffleExchangeError::CursorAhead {
                after,
                current: self.cursor,
            });
        }

        let Some(entries) = self.partition_index.get(&output_partition_id) else {
            return Ok(Vec::new());
        };
        let first_new = entries.partition_point(|(cursor, _)| *cursor <= after);
        Ok(entries[first_new..]
            .iter()
            .map(|(_, key)| {
                self.artifacts
                    .get(key)
                    .expect("shuffle partition index must reference a committed artifact")
            })
            .collect())
    }

    fn roll_epoch(
        &mut self,
        invalidated_tasks: BTreeSet<usize>,
        force_rollover: bool,
    ) -> Result<ShuffleInvalidationResult> {
        if invalidated_tasks.is_empty() && !force_rollover {
            return Ok(ShuffleInvalidationResult::Unchanged);
        }

        // Epoch overflow is checked before replacement state is built, so the
        // operation is fail-closed and leaves the old history intact.
        let previous_epoch = self.cursor.epoch();
        let current_epoch = previous_epoch
            .checked_next()
            .ok_or(ShuffleExchangeError::EpochExhausted)?;

        let mut retained = Vec::with_capacity(self.artifacts.len());
        for (key, artifact) in &self.artifacts {
            if !invalidated_tasks.contains(&key.producer_task_id) {
                retained.push((*key, artifact.location.clone()));
            }
        }
        retained.sort_by_key(|(key, _)| *key);

        let initial_cursor = ShuffleExchangeCursor::initial(current_epoch);
        let bootstrap_cursor = if retained.is_empty() {
            None
        } else {
            Some(
                initial_cursor
                    .checked_next()
                    .expect("initial shuffle exchange cursor must advance"),
            )
        };
        let cursor = bootstrap_cursor.unwrap_or(initial_cursor);

        let mut artifacts = HashMap::with_capacity(retained.len());
        let mut partition_index:
            HashMap<usize, Vec<(ShuffleExchangeCursor, ShuffleArtifactKey)>> =
            HashMap::new();

        for (key, location) in retained {
            artifacts.insert(
                key,
                PublishedShuffleArtifact {
                    cursor,
                    location,
                },
            );
            partition_index
                .entry(key.output_partition_id)
                .or_default()
                .push((cursor, key));
        }
        for entries in partition_index.values_mut() {
            entries.sort_by_key(|(_, key)| *key);
        }

        let mut accepted_tasks = HashMap::with_capacity(self.accepted_tasks.len());
        for (task, keys) in &self.accepted_tasks {
            if !invalidated_tasks.contains(task) {
                accepted_tasks.insert(*task, keys.clone());
            }
        }

        let rollover = ShuffleEpochRollover {
            previous_epoch,
            current_epoch,
            bootstrap_cursor,
            invalidated_tasks: invalidated_tasks.iter().copied().collect(),
            retained_task_count: accepted_tasks.len(),
            retained_artifact_count: artifacts.len(),
        };

        self.cursor = cursor;
        self.lifecycle = ShuffleExchangeLifecycle::Open;
        self.artifacts = artifacts;
        self.partition_index = partition_index;
        self.accepted_tasks = accepted_tasks;

        Ok(ShuffleInvalidationResult::Rolled(rollover))
    }

    fn ensure_epoch(&self, epoch: ShuffleExchangeEpoch) -> Result<()> {
        let current = self.cursor.epoch();
        if epoch == current {
            Ok(())
        } else {
            Err(ShuffleExchangeError::StaleEpoch {
                requested: epoch,
                current,
            })
        }
    }

    fn canonical_publication(
        &self,
        producer_task_id: usize,
        locations: Vec<PartitionLocation>,
    ) -> Result<Vec<PartitionLocation>> {
        let mut by_key = HashMap::with_capacity(locations.len());

        for location in locations {
            let actual_exchange = ShuffleExchangeId::new(
                location.partition_id.job_id.clone(),
                location.partition_id.stage_id,
            );
            if actual_exchange != self.id {
                return Err(ShuffleExchangeError::WrongExchange {
                    expected: self.id.clone(),
                    actual: actual_exchange,
                });
            }
            if location.map_partition_id != producer_task_id {
                return Err(ShuffleExchangeError::WrongProducerTask {
                    expected: producer_task_id,
                    actual: location.map_partition_id,
                });
            }

            let key = ShuffleArtifactKey::from_location(&location);
            if let Some(previous) = by_key.get(&key) {
                if !same_artifact_location(previous, &location) {
                    return Err(ShuffleExchangeError::ConflictingArtifact { key });
                }
            } else {
                by_key.insert(key, location);
            }
        }

        let mut publication: Vec<_> = by_key.into_values().collect();
        publication.sort_by_key(ShuffleArtifactKey::from_location);
        Ok(publication)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ballista_core::JobId;
    use ballista_core::serde::scheduler::{
        ExecutorMetadata, ExecutorOperatingSystemSpecification, ExecutorSpecification,
        PartitionId, PartitionStats,
    };

    fn exchange() -> ShuffleExchangeState {
        ShuffleExchangeState::new(ShuffleExchangeId::new(JobId::from("job"), 7))
    }

    fn cursor(epoch: ShuffleExchangeEpoch, sequence: u64) -> ShuffleExchangeCursor {
        let mut cursor = ShuffleExchangeCursor::initial(epoch);
        for _ in 0..sequence {
            cursor = cursor.checked_next().unwrap();
        }
        cursor
    }

    fn location(
        task: usize,
        output_partition: usize,
        file_id: Option<u64>,
    ) -> PartitionLocation {
        PartitionLocation {
            map_partition_id: task,
            partition_id: PartitionId::new(&JobId::from("job"), 7, output_partition),
            executor_meta: ExecutorMetadata {
                id: format!("executor-{task}"),
                host: "localhost".into(),
                port: 50051,
                grpc_port: 50052,
                specification: ExecutorSpecification::default(),
                os_info: ExecutorOperatingSystemSpecification::default(),
            },
            partition_stats: PartitionStats::new(Some(10), Some(1), Some(100)),
            file_id,
            is_sort_shuffle: true,
        }
    }

    #[test]
    fn task_publication_is_atomic_and_partition_indexed() {
        let mut state = exchange();
        let epoch = state.epoch();
        let task_zero = vec![
            location(0, 0, Some(10)),
            location(0, 1, Some(11)),
            location(0, 0, Some(12)),
        ];

        assert_eq!(
            state.publish(epoch, 0, task_zero.clone()).unwrap(),
            ShufflePublicationResult::Committed {
                event_cursor: Some(cursor(epoch, 1))
            }
        );
        assert_eq!(state.sequence().get(), 1);
        assert_eq!(state.cursor(), cursor(epoch, 1));
        assert!(state.is_task_accepted(0));
        assert!(state.has_committed_artifacts());
        assert_eq!(state.id().producer_stage_id(), 7);

        let partition_zero = state
            .artifacts_after(cursor(epoch, 0), 0)
            .unwrap();
        assert_eq!(partition_zero.len(), 2);
        assert!(partition_zero.iter().all(|artifact| artifact.cursor().sequence().get() == 1));
        assert_eq!(partition_zero[0].location().file_id, Some(10));
        assert_eq!(partition_zero[1].location().file_id, Some(12));

        let partition_one = state
            .artifacts_after(cursor(epoch, 0), 1)
            .unwrap();
        assert_eq!(partition_one.len(), 1);
        assert!(same_artifact_location(
            partition_one[0].location(),
            &task_zero[1]
        ));
    }

    #[test]
    fn exact_replay_is_idempotent_before_and_after_seal() {
        let mut state = exchange();
        let epoch = state.epoch();
        let first = vec![location(0, 0, Some(10)), location(0, 1, Some(11))];

        state.publish(epoch, 0, first.clone()).unwrap();
        assert_eq!(
            state
                .publish(epoch, 0, first.iter().cloned().rev().collect())
                .unwrap(),
            ShufflePublicationResult::Replay
        );
        assert_eq!(state.sequence().get(), 1);

        // Executor resource/OS metadata is not part of artifact identity.
        let mut metadata_refresh = first.clone();
        metadata_refresh[0].executor_meta.specification.vcores = 64;
        metadata_refresh[0].executor_meta.os_info.total_available_disk_space += 1;
        assert_eq!(
            state.publish(epoch, 0, metadata_refresh).unwrap(),
            ShufflePublicationResult::Replay
        );
        assert_eq!(state.sequence().get(), 1);

        let mut conflict = first.clone();
        conflict[0].executor_meta.id = "changed-executor".into();
        assert_eq!(
            state.publish(epoch, 0, conflict).unwrap_err(),
            ShuffleExchangeError::ConflictingTaskReplay {
                producer_task_id: 0
            }
        );
        assert_eq!(state.sequence().get(), 1);

        state
            .publish(epoch, 1, vec![location(1, 0, Some(20))])
            .unwrap();
        assert_eq!(state.sequence().get(), 2);

        assert_eq!(
            state.publish(epoch, 0, first.clone()).unwrap(),
            ShufflePublicationResult::Replay
        );
        assert_eq!(state.sequence().get(), 2);

        assert_eq!(state.seal(epoch).unwrap().map(|c| c.sequence().get()), Some(3));
        assert_eq!(state.lifecycle(), ShuffleExchangeLifecycle::Sealed);
        assert_eq!(
            state.publish(epoch, 0, first).unwrap(),
            ShufflePublicationResult::Replay
        );
        assert_eq!(state.sequence().get(), 3);

        assert_eq!(
            state
                .publish(epoch, 2, vec![location(2, 0, Some(30))])
                .unwrap_err(),
            ShuffleExchangeError::PublicationAfterSeal
        );
        assert_eq!(
            state.publish(epoch, 3, vec![]).unwrap_err(),
            ShuffleExchangeError::PublicationAfterSeal
        );
    }

    #[test]
    fn empty_publication_is_accepted_without_reader_visible_event() {
        let mut state = exchange();
        let epoch = state.epoch();

        assert_eq!(
            state.publish(epoch, 0, vec![]).unwrap(),
            ShufflePublicationResult::Committed {
                event_cursor: None
            }
        );
        assert!(state.is_task_accepted(0));
        assert!(!state.has_committed_artifacts());
        assert_eq!(state.sequence(), ShuffleExchangeSequence::INITIAL);
        assert_eq!(
            state.publish(epoch, 0, vec![]).unwrap(),
            ShufflePublicationResult::Replay
        );

        assert_eq!(
            state
                .publish(epoch, 0, vec![location(0, 0, Some(1))])
                .unwrap_err(),
            ShuffleExchangeError::ConflictingTaskReplay {
                producer_task_id: 0
            }
        );
        assert_eq!(state.sequence(), ShuffleExchangeSequence::INITIAL);
    }

    #[test]
    fn conflicting_duplicates_fail_without_partial_publication() {
        let mut state = exchange();
        let epoch = state.epoch();
        let first = location(0, 0, Some(10));
        let mut conflict = first.clone();
        conflict.executor_meta.id = "different-executor".into();

        assert_eq!(
            state.publish(epoch, 0, vec![first, conflict]).unwrap_err(),
            ShuffleExchangeError::ConflictingArtifact {
                key: ShuffleArtifactKey {
                    producer_task_id: 0,
                    output_partition_id: 0,
                    file_id: Some(10),
                }
            }
        );
        assert_eq!(state.sequence(), ShuffleExchangeSequence::INITIAL);
        assert!(!state.is_task_accepted(0));
        assert!(!state.has_committed_artifacts());
    }

    #[test]
    fn exact_duplicates_inside_one_report_are_canonicalized() {
        let mut state = exchange();
        let epoch = state.epoch();
        let artifact = location(0, 0, Some(10));

        state
            .publish(epoch, 0, vec![artifact.clone(), artifact.clone()])
            .unwrap();

        let visible = state
            .artifacts_after(cursor(epoch, 0), 0)
            .unwrap();
        assert_eq!(visible.len(), 1);
        assert!(same_artifact_location(visible[0].location(), &artifact));
    }

    #[test]
    fn malformed_identity_fails_atomically() {
        let mut state = exchange();
        let epoch = state.epoch();

        let valid = location(0, 0, Some(9));
        let mut wrong_exchange = location(0, 1, Some(10));
        wrong_exchange.partition_id.stage_id = 8;
        assert!(matches!(
            state.publish(epoch, 0, vec![valid, wrong_exchange]),
            Err(ShuffleExchangeError::WrongExchange { .. })
        ));

        let wrong_task = location(1, 0, Some(11));
        assert_eq!(
            state.publish(epoch, 0, vec![wrong_task]).unwrap_err(),
            ShuffleExchangeError::WrongProducerTask {
                expected: 0,
                actual: 1,
            }
        );

        assert_eq!(state.sequence(), ShuffleExchangeSequence::INITIAL);
        assert!(!state.is_task_accepted(0));
        assert!(!state.has_committed_artifacts());
    }

    #[test]
    fn partition_reads_return_only_events_after_cursor() {
        let mut state = exchange();
        let epoch = state.epoch();

        state
            .publish(epoch, 0, vec![location(0, 0, Some(10))])
            .unwrap();
        state
            .publish(
                epoch,
                1,
                vec![location(1, 0, Some(20)), location(1, 1, Some(21))],
            )
            .unwrap();

        let delta = state
            .artifacts_after(cursor(epoch, 1), 0)
            .unwrap();
        assert_eq!(delta.len(), 1);
        assert_eq!(delta[0].cursor().sequence().get(), 2);
        assert_eq!(delta[0].location().map_partition_id, 1);

        assert!(state
            .artifacts_after(cursor(epoch, 2), 1)
            .unwrap()
            .is_empty());

        assert_eq!(
            state
                .artifacts_after(cursor(epoch, 3), 0)
                .unwrap_err(),
            ShuffleExchangeError::CursorAhead {
                after: cursor(epoch, 3),
                current: cursor(epoch, 2),
            }
        );
    }

    #[test]
    fn stale_epoch_is_rejected_without_mutation() {
        let mut state = exchange();
        let stale = ShuffleExchangeEpoch::new(2).unwrap();

        assert_eq!(
            state
                .publish(stale, 0, vec![location(0, 0, Some(10))])
                .unwrap_err(),
            ShuffleExchangeError::StaleEpoch {
                requested: stale,
                current: ShuffleExchangeEpoch::INITIAL,
            }
        );
        assert_eq!(state.sequence(), ShuffleExchangeSequence::INITIAL);
        assert!(!state.is_task_accepted(0));

        assert_eq!(
            state
                .artifacts_after(ShuffleExchangeCursor::initial(stale), 0)
                .unwrap_err(),
            ShuffleExchangeError::StaleEpoch {
                requested: stale,
                current: ShuffleExchangeEpoch::INITIAL,
            }
        );
    }

    #[test]
    fn task_invalidation_rolls_epoch_and_rebases_survivors() {
        let mut state = exchange();
        let epoch_one = state.epoch();
        let task_zero = vec![location(0, 0, Some(10)), location(0, 1, Some(11))];
        let task_one = vec![location(1, 0, Some(20))];

        state.publish(epoch_one, 0, task_zero.clone()).unwrap();
        state.publish(epoch_one, 1, task_one.clone()).unwrap();

        let result = state.invalidate_tasks(epoch_one, &[0]).unwrap();
        let ShuffleInvalidationResult::Rolled(rollover) = result else {
            panic!("expected epoch rollover")
        };
        let epoch_two = rollover.current_epoch;

        assert_eq!(rollover.previous_epoch, epoch_one);
        assert_eq!(epoch_two.get(), 2);
        assert_eq!(
            rollover.bootstrap_cursor,
            Some(cursor(epoch_two, 1))
        );
        assert_eq!(rollover.invalidated_tasks, vec![0]);
        assert_eq!(rollover.retained_task_count, 1);
        assert_eq!(rollover.retained_artifact_count, 1);
        assert_eq!(state.cursor(), cursor(epoch_two, 1));
        assert_eq!(state.lifecycle(), ShuffleExchangeLifecycle::Open);
        assert!(!state.is_task_accepted(0));
        assert!(state.is_task_accepted(1));

        assert!(matches!(
            state.artifacts_after(ShuffleExchangeCursor::initial(epoch_one), 0),
            Err(ShuffleExchangeError::StaleEpoch { .. })
        ));

        // Surviving publication remains an exact replay in the new epoch.
        assert_eq!(
            state.publish(epoch_two, 1, task_one).unwrap(),
            ShufflePublicationResult::Replay
        );

        // Invalidated output can be replaced.
        assert_eq!(
            state.publish(epoch_two, 0, task_zero).unwrap(),
            ShufflePublicationResult::Committed {
                event_cursor: Some(cursor(epoch_two, 2))
            }
        );

        // A task that never materialized output in epoch one may finish after
        // rollover and commit into the current epoch once scheduler task-state
        // validation says its attempt is still authorized.
        assert_eq!(
            state
                .publish(epoch_two, 2, vec![location(2, 0, Some(30))])
                .unwrap(),
            ShufflePublicationResult::Committed {
                event_cursor: Some(cursor(epoch_two, 3))
            }
        );
    }

    #[test]
    fn artifact_loss_invalidates_whole_producer_task() {
        let mut state = exchange();
        let epoch_one = state.epoch();
        let task_zero = vec![location(0, 0, Some(10)), location(0, 1, Some(11))];

        state.publish(epoch_one, 0, task_zero.clone()).unwrap();
        state
            .publish(epoch_one, 1, vec![location(1, 0, Some(20))])
            .unwrap();

        let lost = ShuffleArtifactKey::from_location(&task_zero[0]);
        let result = state.invalidate_artifacts(epoch_one, &[lost]).unwrap();
        let ShuffleInvalidationResult::Rolled(rollover) = result else {
            panic!("expected epoch rollover")
        };

        assert_eq!(rollover.invalidated_tasks, vec![0]);
        assert!(!state.is_task_accepted(0));
        assert!(state.is_task_accepted(1));
    }

    #[test]
    fn invalidate_all_fences_every_accepted_publication_once() {
        let mut state = exchange();
        let epoch_one = state.epoch();

        state
            .publish(epoch_one, 0, vec![location(0, 0, Some(10))])
            .unwrap();
        state.publish(epoch_one, 1, vec![]).unwrap();
        state
            .publish(epoch_one, 2, vec![location(2, 1, Some(20))])
            .unwrap();

        let result = state.invalidate_all(epoch_one).unwrap();
        let ShuffleInvalidationResult::Rolled(rollover) = result else {
            panic!("expected epoch rollover")
        };

        assert_eq!(rollover.invalidated_tasks, vec![0, 1, 2]);
        assert_eq!(rollover.retained_task_count, 0);
        assert_eq!(rollover.retained_artifact_count, 0);
        assert_eq!(rollover.bootstrap_cursor, None);
        assert_eq!(state.epoch().get(), 2);
        assert_eq!(
            state.cursor(),
            ShuffleExchangeCursor::initial(state.epoch())
        );
        assert_eq!(state.lifecycle(), ShuffleExchangeLifecycle::Open);
        assert!(!state.has_committed_artifacts());
        assert!(!state.is_task_accepted(0));
        assert!(!state.is_task_accepted(1));
        assert!(!state.is_task_accepted(2));
    }

    #[test]
    fn invalidate_all_without_output_still_retires_the_generation() {
        let mut state = exchange();
        let epoch_one = state.epoch();

        let result = state.invalidate_all(epoch_one).unwrap();
        let ShuffleInvalidationResult::Rolled(rollover) = result else {
            panic!("expected forced epoch rollover")
        };

        let epoch_two = state.epoch();
        assert_eq!(rollover.previous_epoch, epoch_one);
        assert_eq!(rollover.current_epoch, epoch_two);
        assert!(epoch_two > epoch_one);
        assert!(rollover.invalidated_tasks.is_empty());
        assert_eq!(rollover.retained_task_count, 0);
        assert_eq!(rollover.retained_artifact_count, 0);
        assert_eq!(rollover.bootstrap_cursor, None);
        assert_eq!(
            state.cursor(),
            ShuffleExchangeCursor::initial(epoch_two)
        );
        assert_eq!(state.lifecycle(), ShuffleExchangeLifecycle::Open);
    }

    #[test]
    fn unknown_and_duplicate_loss_signals_do_not_churn_epoch() {
        let mut state = exchange();
        let epoch = state.epoch();
        state
            .publish(epoch, 0, vec![location(0, 0, Some(10))])
            .unwrap();

        assert_eq!(
            state.invalidate_tasks(epoch, &[99, 99]).unwrap(),
            ShuffleInvalidationResult::Unchanged
        );
        assert_eq!(state.epoch(), epoch);

        let unknown = ShuffleArtifactKey {
            producer_task_id: 0,
            output_partition_id: 0,
            file_id: Some(999),
        };
        assert_eq!(
            state.invalidate_artifacts(epoch, &[unknown]).unwrap(),
            ShuffleInvalidationResult::Unchanged
        );
        assert_eq!(state.epoch(), epoch);
    }

    #[test]
    fn unknown_specific_loss_is_noop_but_full_attempt_reset_rolls() {
        let mut state = exchange();
        let epoch_one = state.epoch();

        assert_eq!(
            state.invalidate_tasks(epoch_one, &[42]).unwrap(),
            ShuffleInvalidationResult::Unchanged
        );
        assert_eq!(state.epoch(), epoch_one);

        assert!(matches!(
            state.invalidate_all(epoch_one).unwrap(),
            ShuffleInvalidationResult::Rolled(_)
        ));
        assert!(state.epoch() > epoch_one);
    }

    #[test]
    fn invalidation_reopens_sealed_exchange_and_stale_epoch_fails_closed() {
        let mut state = exchange();
        let epoch_one = state.epoch();
        state
            .publish(epoch_one, 0, vec![location(0, 0, Some(10))])
            .unwrap();
        state.seal(epoch_one).unwrap();

        state.invalidate_tasks(epoch_one, &[0]).unwrap();
        let epoch_two = state.epoch();

        assert_eq!(state.lifecycle(), ShuffleExchangeLifecycle::Open);
        assert!(epoch_two > epoch_one);
        assert!(matches!(
            state.invalidate_all(epoch_one),
            Err(ShuffleExchangeError::StaleEpoch { .. })
        ));
        assert_eq!(state.epoch(), epoch_two);
    }

    #[test]
    fn epoch_exhaustion_leaves_previous_state_intact() {
        let mut state = exchange();
        let epoch = ShuffleExchangeEpoch::new(u64::MAX).unwrap();
        state.cursor = ShuffleExchangeCursor::initial(epoch);
        state
            .publish(epoch, 0, vec![location(0, 0, Some(10))])
            .unwrap();
        let before = state.cursor();

        assert_eq!(
            state.invalidate_all(epoch).unwrap_err(),
            ShuffleExchangeError::EpochExhausted
        );
        assert_eq!(state.cursor(), before);
        assert!(state.is_task_accepted(0));
        assert!(state.has_committed_artifacts());
    }

    #[test]
    fn seal_is_idempotent() {
        let mut state = exchange();
        let epoch = state.epoch();

        assert_eq!(state.seal(epoch).unwrap().map(|c| c.sequence().get()), Some(1));
        assert_eq!(state.seal(epoch).unwrap(), None);
        assert_eq!(state.sequence().get(), 1);
        assert_eq!(state.lifecycle(), ShuffleExchangeLifecycle::Sealed);
    }
}
