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

use std::collections::HashMap;
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
    fn from_location(location: &PartitionLocation) -> Self {
        Self {
            producer_task_id: location.map_partition_id,
            output_partition_id: location.partition_id.partition_id,
            file_id: location.file_id,
        }
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
/// This PR models only the first epoch. Epoch rollover is intentionally added by
/// the recovery follow-up so invalidation semantics stay independently reviewable.
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
    // Producer task IDs are unique within an epoch. Any recovery transition
    // that can reset or reuse the task-ID namespace must roll the exchange
    // epoch before accepting replacement publications.
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
    /// same event sequence.
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
    fn seal_is_idempotent() {
        let mut state = exchange();
        let epoch = state.epoch();

        assert_eq!(state.seal(epoch).unwrap().map(|c| c.sequence().get()), Some(1));
        assert_eq!(state.seal(epoch).unwrap(), None);
        assert_eq!(state.sequence().get(), 1);
        assert_eq!(state.lifecycle(), ShuffleExchangeLifecycle::Sealed);
    }
}
