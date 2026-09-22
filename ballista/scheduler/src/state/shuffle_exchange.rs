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
use std::sync::Arc;

use ballista_core::serde::scheduler::{ExecutorMetadata, PartitionLocation, ShuffleLayout};
use ballista_core::shuffle::{
    ShuffleExchangeCursor, ShuffleExchangeEpoch, ShuffleExchangeGeneration,
    ShuffleExchangeId, ShuffleExchangeLifecycle, ShuffleExchangeSequence,
};

/// Producer-ownership identity of one immutable shuffle artifact.
///
/// This key answers which task publication owns the artifact. Reader-facing
/// fetch identity is tracked separately so ownership cannot hide two logical
/// artifacts aliasing the same concrete shuffle address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ShuffleArtifactKey {
    producer_task_id: usize,
    output_partition_id: usize,
}

impl ShuffleArtifactKey {
    /// Builds the logical artifact ownership key from one validated location.
    fn from_location(location: &PartitionLocation) -> Self {
        Self {
            producer_task_id: location.map_partition_id,
            output_partition_id: location.partition_id.partition_id,
        }
    }
}

/// Exchange-local identity of one physical shuffle data-file path.
///
/// This models the relative path below job/stage exactly. Both sort file 10 and
/// passthrough partition 10 without a file id resolve to 10/data.arrow and must
/// therefore compare equal even though their layouts differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ShuffleFileKey {
    directory_id: u128,
    file_name: ShuffleFileName,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum ShuffleFileName {
    Default,
    Numbered(u64),
}

impl ShuffleFileKey {
    /// Returns the passthrough/range physical file identity.
    pub(crate) fn passthrough(
        output_partition_id: usize,
        file_id: Option<u64>,
    ) -> Self {
        Self {
            directory_id: output_partition_id as u128,
            file_name: file_id
                .map(ShuffleFileName::Numbered)
                .unwrap_or(ShuffleFileName::Default),
        }
    }

    /// Returns the consolidated sort-shuffle physical file identity.
    pub(crate) fn sort(file_id: u64) -> Self {
        Self {
            directory_id: u128::from(file_id),
            file_name: ShuffleFileName::Default,
        }
    }

    /// Resolves a reported location to its concrete exchange-local data-file path.
    fn from_location(location: &PartitionLocation) -> Option<Self> {
        if location.is_sort_shuffle {
            location.file_id.map(Self::sort)
        } else {
            Some(Self::passthrough(
                location.partition_id.partition_id,
                location.file_id,
            ))
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ShuffleDataEndpoint {
    executor_id: String,
    host: String,
    port: u16,
}

impl ShuffleDataEndpoint {
    /// Captures only executor metadata that changes the shuffle data endpoint.
    fn from_executor(executor: &ExecutorMetadata) -> Self {
        Self {
            executor_id: executor.id.clone(),
            host: executor.host.clone(),
            port: executor.port,
        }
    }

    /// Extracts the producer data endpoint carried by a partition location.
    fn from_location(location: &PartitionLocation) -> Self {
        Self::from_executor(&location.executor_meta)
    }
}

/// Scheduler-owned authority for one producer task to publish into this exchange.
///
/// This capability binds the logical exchange generation, task identity, data
/// endpoint, scheduler-authorized output set, and the exact physical paths that
/// current writers are allowed to own. Executor reports never supply these
/// fields back to ShuffleExchangeState as independent trust inputs.
#[derive(Clone, Debug)]
pub(crate) struct ShuffleTaskAuthorization {
    generation: ShuffleExchangeGeneration,
    producer_task_id: usize,
    endpoint: ShuffleDataEndpoint,
    expected_output_partitions: Vec<usize>,
    physical_files: Vec<ShuffleFileKey>,
}

/// Executor-reported completion payload for one authorized producer task.
///
/// reported_output_partitions is an authoritative completion manifest, not a
/// reader-visible artifact list. Sort shuffle reports every output bucket in
/// the manifest even when a bucket contains zero rows; locations may remain
/// sparse and contain only reader-visible artifacts.
#[derive(Clone, Debug)]
pub(crate) struct ShuffleTaskPublication {
    reported_output_partitions: Vec<usize>,
    reader_visible_output_partitions: Vec<usize>,
    locations: Vec<PartitionLocation>,
}

impl ShuffleTaskPublication {
    /// Builds one executor completion report.
    ///
    /// reader_visible_output_partitions must be derived from the unfiltered
    /// executor summaries. For sort shuffle it is the set whose summaries have
    /// rows and therefore require PartitionLocations; for passthrough it is the
    /// complete reported output set.
    pub(crate) fn new(
        reported_output_partitions: Vec<usize>,
        reader_visible_output_partitions: Vec<usize>,
        locations: Vec<PartitionLocation>,
    ) -> Self {
        Self {
            reported_output_partitions,
            reader_visible_output_partitions,
            locations,
        }
    }
}

/// Compares immutable shuffle semantics while ignoring unrelated executor metadata.
fn same_artifact_location(left: &PartitionLocation, right: &PartitionLocation) -> bool {
    left.map_partition_id == right.map_partition_id
        && left.partition_id == right.partition_id
        && left.executor_meta.id == right.executor_meta.id
        && left.executor_meta.host == right.executor_meta.host
        && left.executor_meta.port == right.executor_meta.port
        && left.partition_stats == right.partition_stats
        && left.file_id == right.file_id
        && left.is_sort_shuffle == right.is_sort_shuffle
}

/// One committed artifact together with the exchange event that exposed it.
#[derive(Clone, Debug)]
pub(crate) struct PublishedShuffleArtifact {
    sequence: ShuffleExchangeSequence,
    location: PartitionLocation,
}

impl PublishedShuffleArtifact {
    /// Returns the exchange event sequence that exposed this artifact.
    ///
    /// The owning [ShuffleExchangeState] supplies the generation when a full
    /// reader cursor is needed; storing it once per artifact would duplicate the
    /// job id and exchange identity across potentially many shuffle artifacts.
    pub(crate) fn sequence(&self) -> ShuffleExchangeSequence {
        self.sequence
    }

    /// Returns the immutable artifact location.
    pub(crate) fn location(&self) -> &PartitionLocation {
        &self.location
    }
}

/// Result of accepting a producer task publication.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ShufflePublicationResult {
    /// The task publication was accepted for the first time.
    ///
    /// Empty task output is still recorded as accepted, but has no reader-visible
    /// event and therefore no event cursor.
    Committed {
        event_cursor: Option<ShuffleExchangeCursor>,
    },
    /// The scheduler observed an exact replay of a previously accepted task.
    ///
    /// The original event cursor is returned so a retried acknowledgement has
    /// the same semantic publication identity as the first successful call.
    Replay {
        event_cursor: Option<ShuffleExchangeCursor>,
    },
}

/// Reader position scoped to exactly one output partition.
///
/// A global exchange sequence is meaningful to all partitions, but advancing
/// one partition through that sequence does not mean another partition has
/// consumed the artifacts exposed by the same events. Binding the partition id
/// into the cursor makes that distinction structural instead of a caller
/// convention.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ShufflePartitionCursor {
    exchange_cursor: ShuffleExchangeCursor,
    output_partition_id: usize,
}

impl ShufflePartitionCursor {
    fn new(exchange_cursor: ShuffleExchangeCursor, output_partition_id: usize) -> Self {
        Self {
            exchange_cursor,
            output_partition_id,
        }
    }

    /// Returns the logical exchange this cursor belongs to.
    pub(crate) fn exchange_id(&self) -> &ShuffleExchangeId {
        self.exchange_cursor.exchange_id()
    }

    /// Returns the materialized-history epoch this cursor belongs to.
    pub(crate) fn epoch(&self) -> ShuffleExchangeEpoch {
        self.exchange_cursor.epoch()
    }

    /// Returns the last exchange event scanned for this partition.
    pub(crate) fn sequence(&self) -> ShuffleExchangeSequence {
        self.exchange_cursor.sequence()
    }

    /// Returns the output partition this reader position is bound to.
    pub(crate) fn output_partition_id(&self) -> usize {
        self.output_partition_id
    }
}

/// Atomic owned snapshot of one output partition through a partition-bound cursor.
///
/// A consumer may persist through_cursor only after consuming every artifact
/// returned in this delta. The cursor cannot then be reused for another output
/// partition by accident.
#[derive(Debug)]
pub(crate) struct ShufflePartitionDelta {
    through_cursor: ShufflePartitionCursor,
    lifecycle: ShuffleExchangeLifecycle,
    artifacts: Vec<Arc<PublishedShuffleArtifact>>,
}

impl ShufflePartitionDelta {
    /// Returns the partition-bound cursor through which this output was scanned.
    pub(crate) fn through_cursor(&self) -> &ShufflePartitionCursor {
        &self.through_cursor
    }

    /// Returns the exchange lifecycle observed with this delta snapshot.
    pub(crate) fn lifecycle(&self) -> ShuffleExchangeLifecycle {
        self.lifecycle
    }

    /// Returns owned artifacts published after the caller's cursor through this snapshot.
    pub(crate) fn artifacts(&self) -> &[Arc<PublishedShuffleArtifact>] {
        &self.artifacts
    }
}

/// Structural exchange-state error.
///
/// These variants are intentionally specific so later scheduler integration can
/// decide whether an error is a stale report, a retryable recovery condition, or
/// an internal invariant violation without parsing strings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ShuffleExchangeError {
    /// An operation targeted an epoch older than the current generation.
    StaleEpoch {
        requested: ShuffleExchangeEpoch,
        current: ShuffleExchangeEpoch,
    },
    /// An operation targeted an epoch newer than the current generation.
    FutureEpoch {
        requested: ShuffleExchangeEpoch,
        current: ShuffleExchangeEpoch,
    },
    /// An operation or reported artifact belongs to another logical exchange.
    WrongExchange {
        expected: ShuffleExchangeId,
        actual: ShuffleExchangeId,
    },
    /// A reported artifact names a different producer task.
    WrongProducerTask {
        expected: usize,
        actual: usize,
    },
    /// Two artifacts used the same ownership key but disagreed on contents.
    ConflictingArtifact { key: ShuffleArtifactKey },
    /// Two producer tasks attempted to claim the same physical shuffle file.
    ConflictingFileOwnership {
        file: ShuffleFileKey,
        existing_producer_task_id: usize,
        incoming_producer_task_id: usize,
    },
    /// Sort shuffle requires a file id to address its consolidated data file.
    MissingSortShuffleFileId { producer_task_id: usize },
    /// A reported file id does not match the scheduler task id used by current writers.
    WrongProducerFileId {
        producer_task_id: usize,
        expected: u64,
        actual: Option<u64>,
    },
    /// One task publication claimed artifacts on more than one data endpoint.
    MixedExecutorPublication { producer_task_id: usize },
    /// A publication used a layout different from the exchange's established layout.
    WrongShuffleLayout {
        expected: ShuffleLayout,
        actual: ShuffleLayout,
    },
    /// A reported output partition is outside this exchange's immutable shape.
    InvalidOutputPartition {
        actual: usize,
        partition_count: usize,
    },
    /// A producer report did not cover exactly the output partitions it was authorized for.
    WrongTaskOutputPartitions {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// A task reported an output partition outside its scheduler-authorized set.
    UnexpectedTaskOutputPartition { actual: usize },
    /// A task replay did not exactly match the task publication already accepted.
    ConflictingTaskReplay { producer_task_id: usize },
    /// A previously unseen task attempted to publish after the epoch sealed.
    PublicationAfterSeal,
    /// A reader cursor is ahead of the exchange's current event sequence.
    CursorAhead {
        after: ShuffleExchangeSequence,
        current: ShuffleExchangeSequence,
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
            Self::FutureEpoch { requested, current } => write!(
                f,
                "future shuffle exchange epoch: requested {}, current {}",
                requested.get(),
                current.get()
            ),
            Self::WrongExchange { expected, actual } => write!(
                f,
                "shuffle exchange identity is {}/{}, expected {}/{}",
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
                write!(f, "conflicting shuffle artifact for ownership key {key:?}")
            }
            Self::ConflictingFileOwnership {
                file,
                existing_producer_task_id,
                incoming_producer_task_id,
            } => write!(
                f,
                "shuffle file {file:?} is owned by producer task {existing_producer_task_id}, not {incoming_producer_task_id}"
            ),
            Self::MissingSortShuffleFileId { producer_task_id } => write!(
                f,
                "sort shuffle publication from producer task {producer_task_id} has no file id"
            ),
            Self::WrongProducerFileId {
                producer_task_id,
                expected,
                actual,
            } => write!(
                f,
                "shuffle publication from producer task {producer_task_id} has file id {actual:?}, expected {expected}"
            ),
            Self::MixedExecutorPublication { producer_task_id } => write!(
                f,
                "shuffle publication from producer task {producer_task_id} spans multiple data endpoints"
            ),
            Self::WrongShuffleLayout { expected, actual } => write!(
                f,
                "shuffle publication layout {actual:?} does not match exchange layout {expected:?}"
            ),
            Self::InvalidOutputPartition {
                actual,
                partition_count,
            } => write!(
                f,
                "shuffle output partition {actual} is outside partition count {partition_count}"
            ),
            Self::WrongTaskOutputPartitions { expected, actual } => write!(
                f,
                "shuffle publication output partitions {actual:?} do not match authorized partitions {expected:?}"
            ),
            Self::UnexpectedTaskOutputPartition { actual } => write!(
                f,
                "shuffle publication reported unauthorized output partition {actual}"
            ),
            Self::ConflictingTaskReplay { producer_task_id } => write!(
                f,
                "conflicting replay for producer task {producer_task_id}"
            ),
            Self::PublicationAfterSeal => {
                f.write_str("shuffle publication attempted after exchange seal")
            }
            Self::CursorAhead { after, current } => write!(
                f,
                "shuffle cursor sequence {} is ahead of current sequence {}",
                after.get(),
                current.get()
            ),
            Self::SequenceExhausted => f.write_str("shuffle exchange sequence exhausted"),
        }
    }
}

impl Error for ShuffleExchangeError {}

type Result<T> = std::result::Result<T, ShuffleExchangeError>;

#[derive(Clone, Debug)]
struct AcceptedTaskPublication {
    artifacts: Vec<Arc<PublishedShuffleArtifact>>,
    endpoint: ShuffleDataEndpoint,
    expected_output_partitions: Vec<usize>,
    physical_files: Vec<ShuffleFileKey>,
    event_sequence: Option<ShuffleExchangeSequence>,
}

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
    generation: ShuffleExchangeGeneration,
    output_partition_count: usize,
    layout: ShuffleLayout,
    sequence: ShuffleExchangeSequence,
    lifecycle: ShuffleExchangeLifecycle,
    // Physical shuffle file -> task that first claimed the path in the retained
    // lifetime fence. Recovery must preserve retired identities because the
    // current on-disk path does not contain the exchange epoch. Once a key is
    // present, only the exact replay path above the owner check may reference it;
    // even reuse by the same numeric task id is unsafe after task-id reset.
    file_owners: HashMap<ShuffleFileKey, usize>,
    // Output partition -> append-ordered immutable artifacts. The partition
    // count is fixed, so a Vec avoids hashing partition ids and accepted task
    // publications share the same artifacts via Arc rather than duplicating
    // PartitionLocation metadata.
    partition_index: Vec<Vec<Arc<PublishedShuffleArtifact>>>,
    // Producer task -> canonical committed publication. Physical file claims
    // and the producer data endpoint are retained even when the task produced
    // no logical artifacts, so empty output still fences its on-disk files and
    // has exact replay semantics.
    //
    // Producer task IDs identify accepted task attempts within this epoch.
    // Rolling the epoch fences logical history, but it does not by itself make
    // reuse of an old physical file path safe because the current shuffle path
    // does not encode the epoch. The recovery follow-up must preserve that
    // separate physical-file lifetime invariant.
    accepted_tasks: HashMap<usize, AcceptedTaskPublication>,
}

impl ShuffleExchangeState {
    /// Creates the initial epoch for one logical shuffle exchange.
    pub(crate) fn new(
        id: ShuffleExchangeId,
        output_partition_count: usize,
        layout: ShuffleLayout,
    ) -> Self {
        Self {
            generation: ShuffleExchangeGeneration::initial(id),
            output_partition_count,
            layout,
            sequence: ShuffleExchangeSequence::INITIAL,
            lifecycle: ShuffleExchangeLifecycle::Open,
            file_owners: HashMap::new(),
            partition_index: (0..output_partition_count).map(|_| Vec::new()).collect(),
            accepted_tasks: HashMap::new(),
        }
    }

    /// Returns the logical exchange identity.
    pub(crate) fn id(&self) -> &ShuffleExchangeId {
        self.generation.exchange_id()
    }

    /// Returns the current materialized-history generation.
    pub(crate) fn generation(&self) -> &ShuffleExchangeGeneration {
        &self.generation
    }

    /// Returns the immutable number of reader-visible output partitions.
    pub(crate) fn output_partition_count(&self) -> usize {
        self.output_partition_count
    }

    /// Returns the immutable physical shuffle layout for this exchange.
    pub(crate) fn layout(&self) -> ShuffleLayout {
        self.layout
    }

    /// Returns the current materialized-history epoch.
    pub(crate) fn epoch(&self) -> ShuffleExchangeEpoch {
        self.generation.epoch()
    }

    /// Returns the current generation-bound exchange event position.
    pub(crate) fn cursor(&self) -> ShuffleExchangeCursor {
        self.cursor_at(self.sequence)
    }

    /// Returns the initial reader position for one output partition.
    pub(crate) fn initial_partition_cursor(
        &self,
        output_partition_id: usize,
    ) -> Result<ShufflePartitionCursor> {
        self.ensure_output_partition(output_partition_id)?;
        Ok(ShufflePartitionCursor::new(
            ShuffleExchangeCursor::initial(self.generation.clone()),
            output_partition_id,
        ))
    }

    /// Returns the latest reader-visible event sequence.
    ///
    /// This is primarily useful for diagnostics and tests. Partition reader
    /// APIs use [ShufflePartitionCursor] so progress cannot be detached from
    /// either the generation or the output partition whose history was consumed.
    pub(crate) fn sequence(&self) -> ShuffleExchangeSequence {
        self.sequence
    }

    /// Returns the publication lifecycle of the current epoch.
    pub(crate) fn lifecycle(&self) -> ShuffleExchangeLifecycle {
        self.lifecycle
    }

    /// Returns whether at least one materialized artifact has been committed.
    pub(crate) fn has_committed_artifacts(&self) -> bool {
        self.partition_index.iter().any(|partition| !partition.is_empty())
    }

    /// Returns whether a producer task publication, including empty output, was accepted.
    pub(crate) fn is_task_accepted(&self, producer_task_id: usize) -> bool {
        self.accepted_tasks.contains_key(&producer_task_id)
    }

    /// Builds a generation-bound scheduler authorization for one task.
    ///
    /// The caller must derive expected_output_partitions from scheduler-owned
    /// task state, never from the executor completion report. Sort shuffle is
    /// structurally fixed to the complete exchange output shape because every
    /// sort task hashes into all output buckets. Physical paths are derived here
    /// from that authorization and the current writer contract.
    pub(crate) fn authorize_task(
        &self,
        producer_task_id: usize,
        producer_executor: &ExecutorMetadata,
        mut expected_output_partitions: Vec<usize>,
    ) -> Result<ShuffleTaskAuthorization> {
        expected_output_partitions.sort_unstable();
        expected_output_partitions.dedup();
        for &output_partition_id in &expected_output_partitions {
            self.ensure_output_partition(output_partition_id)?;
        }

        if self.layout == ShuffleLayout::Sort {
            let complete_output_shape = (0..self.output_partition_count).collect::<Vec<_>>();
            if expected_output_partitions != complete_output_shape {
                return Err(ShuffleExchangeError::WrongTaskOutputPartitions {
                    expected: complete_output_shape,
                    actual: expected_output_partitions,
                });
            }
        }

        let expected_file_id = producer_task_id as u64;
        let physical_files = if self.layout == ShuffleLayout::Sort {
            vec![ShuffleFileKey::sort(expected_file_id)]
        } else {
            expected_output_partitions
                .iter()
                .map(|&output_partition_id| {
                    ShuffleFileKey::passthrough(output_partition_id, Some(expected_file_id))
                })
                .collect()
        };

        Ok(ShuffleTaskAuthorization {
            generation: self.generation.clone(),
            producer_task_id,
            endpoint: ShuffleDataEndpoint::from_executor(producer_executor),
            expected_output_partitions,
            physical_files,
        })
    }

    /// Atomically accepts one complete producer-task publication.
    ///
    /// Exact replay is idempotent, including after the exchange is sealed.
    /// Conflicting replay and all malformed publications fail before any state
    /// changes. Scheduler authorization supplies generation, task identity,
    /// producer endpoint, output shape, and physical-file ownership. The
    /// executor report independently supplies a complete output manifest plus
    /// reader-visible locations.
    ///
    /// Sort shuffle may report sparse locations because zero-row buckets are
    /// not reader-visible, but its manifest must still cover the entire
    /// scheduler-authorized output shape. Passthrough locations themselves must
    /// cover the complete manifest because each authorized output has its own
    /// independently addressable file.
    pub(crate) fn publish(
        &mut self,
        authorization: &ShuffleTaskAuthorization,
        report: ShuffleTaskPublication,
    ) -> Result<ShufflePublicationResult> {
        self.ensure_generation(&authorization.generation)?;

        let producer_task_id = authorization.producer_task_id;
        let endpoint = &authorization.endpoint;
        let expected_output_partitions = &authorization.expected_output_partitions;
        let physical_files = &authorization.physical_files;

        let ShuffleTaskPublication {
            mut reported_output_partitions,
            mut reader_visible_output_partitions,
            locations,
        } = report;

        reported_output_partitions.sort_unstable();
        reported_output_partitions.dedup();
        for &output_partition_id in &reported_output_partitions {
            self.ensure_output_partition(output_partition_id)?;
        }
        if reported_output_partitions.as_slice() != expected_output_partitions.as_slice() {
            return Err(ShuffleExchangeError::WrongTaskOutputPartitions {
                expected: expected_output_partitions.to_vec(),
                actual: reported_output_partitions,
            });
        }

        reader_visible_output_partitions.sort_unstable();
        reader_visible_output_partitions.dedup();
        for &output_partition_id in &reader_visible_output_partitions {
            self.ensure_output_partition(output_partition_id)?;
            if expected_output_partitions
                .binary_search(&output_partition_id)
                .is_err()
            {
                return Err(ShuffleExchangeError::UnexpectedTaskOutputPartition {
                    actual: output_partition_id,
                });
            }
        }

        let publication =
            self.canonical_publication(producer_task_id, endpoint, locations)?;
        let mut actual_output_partitions = publication
            .iter()
            .map(|location| location.partition_id.partition_id)
            .collect::<Vec<_>>();
        actual_output_partitions.sort_unstable();
        actual_output_partitions.dedup();

        match self.layout {
            ShuffleLayout::Sort => {
                if actual_output_partitions != reader_visible_output_partitions {
                    return Err(ShuffleExchangeError::WrongTaskOutputPartitions {
                        expected: reader_visible_output_partitions,
                        actual: actual_output_partitions,
                    });
                }
            }
            ShuffleLayout::Passthrough => {
                if reader_visible_output_partitions.as_slice()
                    != expected_output_partitions.as_slice()
                {
                    return Err(ShuffleExchangeError::WrongTaskOutputPartitions {
                        expected: expected_output_partitions.to_vec(),
                        actual: reader_visible_output_partitions,
                    });
                }
                if actual_output_partitions.as_slice() != expected_output_partitions.as_slice() {
                    return Err(ShuffleExchangeError::WrongTaskOutputPartitions {
                        expected: expected_output_partitions.to_vec(),
                        actual: actual_output_partitions,
                    });
                }
            }
        }

        if let Some(previous) = self.accepted_tasks.get(&producer_task_id) {
            let exact_replay = &previous.endpoint == endpoint
                && previous.expected_output_partitions.as_slice()
                    == expected_output_partitions.as_slice()
                && previous.physical_files.as_slice() == physical_files.as_slice()
                && previous.artifacts.len() == publication.len()
                && previous
                    .artifacts
                    .iter()
                    .zip(publication.iter())
                    .all(|(previous_artifact, location)| {
                        same_artifact_location(previous_artifact.location(), location)
                    });

            if exact_replay {
                return Ok(ShufflePublicationResult::Replay {
                    event_cursor: previous
                        .event_sequence
                        .map(|sequence| self.cursor_at(sequence)),
                });
            }
            return Err(ShuffleExchangeError::ConflictingTaskReplay {
                producer_task_id,
            });
        }

        if self.lifecycle.is_sealed() {
            return Err(ShuffleExchangeError::PublicationAfterSeal);
        }

        for file in physical_files {
            if let Some(existing_producer_task_id) = self.file_owners.get(file) {
                return Err(ShuffleExchangeError::ConflictingFileOwnership {
                    file: *file,
                    existing_producer_task_id: *existing_producer_task_id,
                    incoming_producer_task_id: producer_task_id,
                });
            }
        }

        let event_sequence = if publication.is_empty() {
            self.sequence
                .checked_next()
                .ok_or(ShuffleExchangeError::SequenceExhausted)?;
            None
        } else {
            let next_sequence = self
                .sequence
                .checked_next()
                .ok_or(ShuffleExchangeError::SequenceExhausted)?;
            if next_sequence.checked_next().is_none() {
                return Err(ShuffleExchangeError::SequenceExhausted);
            }
            Some(next_sequence)
        };

        for file in physical_files {
            self.file_owners.insert(*file, producer_task_id);
        }

        if publication.is_empty() {
            self.accepted_tasks.insert(
                producer_task_id,
                AcceptedTaskPublication {
                    artifacts: Vec::new(),
                    endpoint: (*endpoint).clone(),
                    expected_output_partitions: expected_output_partitions.to_vec(),
                    physical_files: physical_files.to_vec(),
                    event_sequence: None,
                },
            );
            return Ok(ShufflePublicationResult::Committed {
                event_cursor: None,
            });
        }

        let next_sequence =
            event_sequence.expect("non-empty publication must allocate an event sequence");
        let event_cursor = self.cursor_at(next_sequence);
        let mut published = Vec::with_capacity(publication.len());

        for location in publication {
            let key = ShuffleArtifactKey::from_location(&location);
            let artifact = Arc::new(PublishedShuffleArtifact {
                sequence: next_sequence,
                location,
            });
            self.partition_index[key.output_partition_id].push(artifact.clone());
            published.push(artifact);
        }

        self.accepted_tasks.insert(
            producer_task_id,
            AcceptedTaskPublication {
                artifacts: published,
                endpoint: (*endpoint).clone(),
                expected_output_partitions: expected_output_partitions.to_vec(),
                physical_files: physical_files.to_vec(),
                event_sequence: Some(next_sequence),
            },
        );
        self.sequence = next_sequence;
        Ok(ShufflePublicationResult::Committed {
            event_cursor: Some(event_cursor),
        })
    }

    /// Seals the current epoch.
    ///
    /// The first seal is reader-visible and consumes one event sequence.
    /// Repeated seal calls are idempotent and return None.
    ///
    /// # Caller contract
    ///
    /// The execution graph must only seal after it has authoritative proof that
    /// no currently authorized producer attempt can still publish into this
    /// generation. This state machine deliberately does not infer stage
    /// completion from the publications it has observed.
    pub(crate) fn seal(
        &mut self,
        epoch: ShuffleExchangeEpoch,
    ) -> Result<Option<ShuffleExchangeCursor>> {
        self.ensure_epoch(epoch)?;
        if self.lifecycle.is_sealed() {
            return Ok(None);
        }

        let next_sequence = self
            .sequence
            .checked_next()
            .ok_or(ShuffleExchangeError::SequenceExhausted)?;
        self.lifecycle = ShuffleExchangeLifecycle::Sealed;
        self.sequence = next_sequence;
        Ok(Some(self.cursor_at(next_sequence)))
    }

    /// Returns an atomic delta snapshot for the cursor's output partition.
    ///
    /// The returned artifacts are ordered by publication sequence and then by
    /// canonical artifact key within a task-atomic publication. through_cursor
    /// remains bound to the same output partition, so advancing this reader can
    /// never mark a sibling partition's artifacts as consumed. Lifecycle is
    /// captured in the same snapshot so a seal remains observable even when no
    /// new artifact exists for this partition.
    pub(crate) fn artifacts_after(
        &self,
        after: &ShufflePartitionCursor,
    ) -> Result<ShufflePartitionDelta> {
        self.ensure_cursor_generation(after)?;
        let output_partition_id = after.output_partition_id();
        if after.sequence() > self.sequence {
            return Err(ShuffleExchangeError::CursorAhead {
                after: after.sequence(),
                current: self.sequence,
            });
        }

        let entries = &self.partition_index[output_partition_id];
        let first_new =
            entries.partition_point(|artifact| artifact.sequence() <= after.sequence());
        let artifacts = entries[first_new..].to_vec();

        Ok(ShufflePartitionDelta {
            through_cursor: ShufflePartitionCursor::new(
                self.cursor(),
                output_partition_id,
            ),
            lifecycle: self.lifecycle,
            artifacts,
        })
    }

    /// Builds a generation-bound cursor at an already allocated sequence.
    fn cursor_at(&self, sequence: ShuffleExchangeSequence) -> ShuffleExchangeCursor {
        ShuffleExchangeCursor::new(self.generation.clone(), sequence)
    }

    /// Validates that a partition reader cursor belongs to this exchange generation.
    fn ensure_cursor_generation(&self, cursor: &ShufflePartitionCursor) -> Result<()> {
        self.ensure_output_partition(cursor.output_partition_id())?;
        if cursor.exchange_id() != self.id() {
            return Err(ShuffleExchangeError::WrongExchange {
                expected: self.id().clone(),
                actual: cursor.exchange_id().clone(),
            });
        }
        self.ensure_epoch(cursor.epoch())
    }

    /// Validates the complete logical exchange generation carried by an authorization.
    fn ensure_generation(&self, generation: &ShuffleExchangeGeneration) -> Result<()> {
        if generation.exchange_id() != self.id() {
            return Err(ShuffleExchangeError::WrongExchange {
                expected: self.id().clone(),
                actual: generation.exchange_id().clone(),
            });
        }
        self.ensure_epoch(generation.epoch())
    }

    /// Distinguishes stale, current, and future epochs for structural callers.
    fn ensure_epoch(&self, epoch: ShuffleExchangeEpoch) -> Result<()> {
        let current = self.epoch();
        match epoch.cmp(&current) {
            std::cmp::Ordering::Less => Err(ShuffleExchangeError::StaleEpoch {
                requested: epoch,
                current,
            }),
            std::cmp::Ordering::Equal => Ok(()),
            std::cmp::Ordering::Greater => Err(ShuffleExchangeError::FutureEpoch {
                requested: epoch,
                current,
            }),
        }
    }

    /// Checks an output partition against the immutable exchange shape.
    fn ensure_output_partition(&self, output_partition_id: usize) -> Result<()> {
        if output_partition_id < self.output_partition_count {
            Ok(())
        } else {
            Err(ShuffleExchangeError::InvalidOutputPartition {
                actual: output_partition_id,
                partition_count: self.output_partition_count,
            })
        }
    }

    /// Validates, de-duplicates, and deterministically orders one task report.
    fn canonical_publication(
        &self,
        producer_task_id: usize,
        producer_endpoint: &ShuffleDataEndpoint,
        locations: Vec<PartitionLocation>,
    ) -> Result<Vec<PartitionLocation>> {
        let mut by_key = HashMap::with_capacity(locations.len());

        for location in locations {
            let actual_exchange = ShuffleExchangeId::new(
                location.partition_id.job_id.clone(),
                location.partition_id.stage_id,
            );
            if actual_exchange != *self.id() {
                return Err(ShuffleExchangeError::WrongExchange {
                    expected: self.id().clone(),
                    actual: actual_exchange,
                });
            }
            if location.map_partition_id != producer_task_id {
                return Err(ShuffleExchangeError::WrongProducerTask {
                    expected: producer_task_id,
                    actual: location.map_partition_id,
                });
            }
            self.ensure_output_partition(location.partition_id.partition_id)?;
            if location.is_sort_shuffle && location.file_id.is_none() {
                return Err(ShuffleExchangeError::MissingSortShuffleFileId {
                    producer_task_id,
                });
            }

            let actual_layout = location.layout();
            if self.layout != actual_layout {
                return Err(ShuffleExchangeError::WrongShuffleLayout {
                    expected: self.layout,
                    actual: actual_layout,
                });
            }

            let expected_file_id = producer_task_id as u64;
            if location.file_id != Some(expected_file_id) {
                return Err(ShuffleExchangeError::WrongProducerFileId {
                    producer_task_id,
                    expected: expected_file_id,
                    actual: location.file_id,
                });
            }

            let endpoint = ShuffleDataEndpoint::from_location(&location);
            if &endpoint != producer_endpoint {
                return Err(ShuffleExchangeError::MixedExecutorPublication {
                    producer_task_id,
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

    fn exchange_id() -> ShuffleExchangeId {
        ShuffleExchangeId::new(JobId::from("job"), 7)
    }

    const OUTPUT_PARTITIONS: usize = 4;

    fn exchange() -> ShuffleExchangeState {
        ShuffleExchangeState::new(
            exchange_id(),
            OUTPUT_PARTITIONS,
            ShuffleLayout::Sort,
        )
    }

    fn passthrough_exchange() -> ShuffleExchangeState {
        ShuffleExchangeState::new(
            exchange_id(),
            OUTPUT_PARTITIONS,
            ShuffleLayout::Passthrough,
        )
    }

    fn cursor(epoch: ShuffleExchangeEpoch, sequence: u64) -> ShuffleExchangeCursor {
        ShuffleExchangeCursor::new(
            ShuffleExchangeGeneration::new(exchange_id(), epoch),
            ShuffleExchangeSequence::new(sequence),
        )
    }

    fn partition_cursor(
        epoch: ShuffleExchangeEpoch,
        output_partition_id: usize,
        sequence: u64,
    ) -> ShufflePartitionCursor {
        ShufflePartitionCursor::new(cursor(epoch, sequence), output_partition_id)
    }

    fn cursor_for(
        exchange_id: ShuffleExchangeId,
        epoch: ShuffleExchangeEpoch,
        sequence: u64,
    ) -> ShuffleExchangeCursor {
        ShuffleExchangeCursor::new(
            ShuffleExchangeGeneration::new(exchange_id, epoch),
            ShuffleExchangeSequence::new(sequence),
        )
    }

    fn executor(task: usize) -> ExecutorMetadata {
        ExecutorMetadata {
            id: format!("executor-{task}"),
            host: "localhost".into(),
            port: 50051,
            grpc_port: 50052,
            specification: ExecutorSpecification::default(),
            os_info: ExecutorOperatingSystemSpecification::default(),
        }
    }

    // Presence-only helper: current writers always stamp the scheduler task id
    // as file_id, so tests pass Some(_) only to request a present file id.
    fn location(
        task: usize,
        output_partition: usize,
        file_id_marker: Option<u64>,
    ) -> PartitionLocation {
        PartitionLocation {
            map_partition_id: task,
            partition_id: PartitionId::new(&JobId::from("job"), 7, output_partition),
            executor_meta: executor(task),
            partition_stats: PartitionStats::new(Some(10), Some(1), Some(100)),
            file_id: file_id_marker.map(|_| task as u64),
            is_sort_shuffle: true,
        }
    }

    fn passthrough_location(
        task: usize,
        output_partition: usize,
        file_id: Option<u64>,
    ) -> PartitionLocation {
        let mut location = location(task, output_partition, file_id);
        location.is_sort_shuffle = false;
        location
    }

    fn publish(
        state: &mut ShuffleExchangeState,
        epoch: ShuffleExchangeEpoch,
        producer_task_id: usize,
        locations: Vec<PartitionLocation>,
    ) -> Result<ShufflePublicationResult> {
        let producer_executor = locations
            .first()
            .map(|location| location.executor_meta.clone())
            .unwrap_or_else(|| executor(producer_task_id));

        let expected_output_partitions = if state.layout() == ShuffleLayout::Sort {
            (0..OUTPUT_PARTITIONS).collect::<Vec<_>>()
        } else {
            let mut output_partitions = locations
                .iter()
                .map(|location| location.partition_id.partition_id)
                .collect::<Vec<_>>();
            output_partitions.sort_unstable();
            output_partitions.dedup();
            output_partitions
        };
        let reported_output_partitions = if state.layout() == ShuffleLayout::Sort {
            (0..OUTPUT_PARTITIONS).collect()
        } else {
            expected_output_partitions.clone()
        };
        let mut reader_visible_output_partitions = locations
            .iter()
            .map(|location| location.partition_id.partition_id)
            .collect::<Vec<_>>();
        reader_visible_output_partitions.sort_unstable();
        reader_visible_output_partitions.dedup();

        let mut authorization = state.authorize_task(
            producer_task_id,
            &producer_executor,
            expected_output_partitions,
        )?;
        authorization.generation =
            ShuffleExchangeGeneration::new(state.id().clone(), epoch);

        state.publish(
            &authorization,
            ShuffleTaskPublication::new(
                reported_output_partitions,
                reader_visible_output_partitions,
                locations,
            ),
        )
    }

    #[test]
    fn task_publication_is_atomic_and_partition_indexed() {
        let mut state = exchange();
        let epoch = state.epoch();
        let task_zero = vec![
            location(0, 0, Some(0)),
            location(0, 1, Some(0)),
            location(0, 2, Some(0)),
        ];

        assert_eq!(
            publish(&mut state, epoch, 0, task_zero.clone()).unwrap(),
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
            .artifacts_after(&partition_cursor(epoch, 0, 0))
            .unwrap();
        assert_eq!(
            partition_zero.through_cursor(),
            &partition_cursor(epoch, 0, 1)
        );
        assert_eq!(partition_zero.lifecycle(), ShuffleExchangeLifecycle::Open);
        assert_eq!(partition_zero.artifacts().len(), 1);
        assert_eq!(partition_zero.artifacts()[0].sequence().get(), 1);
        assert_eq!(partition_zero.artifacts()[0].location().file_id, Some(0));

        let partition_one = state
            .artifacts_after(&partition_cursor(epoch, 1, 0))
            .unwrap();
        assert_eq!(partition_one.artifacts().len(), 1);
        assert!(same_artifact_location(
            partition_one.artifacts()[0].location(),
            &task_zero[1]
        ));
    }

    #[test]
    fn exact_replay_is_idempotent_before_and_after_seal() {
        let mut state = exchange();
        let epoch = state.epoch();
        let first = vec![location(0, 0, Some(10)), location(0, 1, Some(11))];

        publish(&mut state, epoch, 0, first.clone()).unwrap();
        assert_eq!(
            publish(&mut state, epoch, 0, first.iter().cloned().rev().collect())
                .unwrap(),
            ShufflePublicationResult::Replay {
                event_cursor: Some(cursor(epoch, 1))
            }
        );
        assert_eq!(state.sequence().get(), 1);

        // Executor resource/OS metadata is not part of artifact identity.
        let mut metadata_refresh = first.clone();
        metadata_refresh[0].executor_meta.specification.vcores = 64;
        metadata_refresh[0].executor_meta.os_info.total_available_disk_space += 1;
        metadata_refresh[0].executor_meta.grpc_port += 1;
        assert_eq!(
            publish(&mut state, epoch, 0, metadata_refresh).unwrap(),
            ShufflePublicationResult::Replay {
                event_cursor: Some(cursor(epoch, 1))
            }
        );
        assert_eq!(state.sequence().get(), 1);

        let mut conflict = first.clone();
        for location in &mut conflict {
            location.executor_meta.id = "changed-executor".into();
        }
        assert_eq!(
            publish(&mut state, epoch, 0, conflict).unwrap_err(),
            ShuffleExchangeError::ConflictingTaskReplay {
                producer_task_id: 0
            }
        );
        assert_eq!(state.sequence().get(), 1);

        publish(&mut state, epoch, 1, vec![location(1, 0, Some(20))])
            .unwrap();
        assert_eq!(state.sequence().get(), 2);

        assert_eq!(
            publish(&mut state, epoch, 0, first.clone()).unwrap(),
            ShufflePublicationResult::Replay {
                event_cursor: Some(cursor(epoch, 1))
            }
        );
        assert_eq!(state.sequence().get(), 2);

        assert_eq!(state.seal(epoch).unwrap().map(|c| c.sequence().get()), Some(3));
        assert_eq!(state.lifecycle(), ShuffleExchangeLifecycle::Sealed);
        assert_eq!(
            publish(&mut state, epoch, 0, first).unwrap(),
            ShufflePublicationResult::Replay {
                event_cursor: Some(cursor(epoch, 1))
            }
        );
        assert_eq!(state.sequence().get(), 3);

        assert_eq!(
            publish(&mut state, epoch, 2, vec![location(2, 0, Some(30))])
                .unwrap_err(),
            ShuffleExchangeError::PublicationAfterSeal
        );
        assert_eq!(
            publish(&mut state, epoch, 3, vec![]).unwrap_err(),
            ShuffleExchangeError::PublicationAfterSeal
        );
    }

    #[test]
    fn empty_publication_is_accepted_without_reader_visible_event() {
        let mut state = exchange();
        let epoch = state.epoch();

        assert_eq!(
            publish(&mut state, epoch, 0, vec![]).unwrap(),
            ShufflePublicationResult::Committed {
                event_cursor: None
            }
        );
        assert!(state.is_task_accepted(0));
        assert!(!state.has_committed_artifacts());
        assert_eq!(state.sequence(), ShuffleExchangeSequence::INITIAL);
        assert_eq!(
            publish(&mut state, epoch, 0, vec![]).unwrap(),
            ShufflePublicationResult::Replay { event_cursor: None }
        );

        assert_eq!(
            publish(&mut state, epoch, 0, vec![location(0, 0, Some(1))])
                .unwrap_err(),
            ShuffleExchangeError::ConflictingTaskReplay {
                producer_task_id: 0
            }
        );
        assert_eq!(state.sequence(), ShuffleExchangeSequence::INITIAL);
    }

    #[test]
    fn empty_sort_publication_reserves_derived_physical_file_and_endpoint() {
        let mut state = exchange();
        let producer = executor(0);
        let expected = (0..OUTPUT_PARTITIONS).collect::<Vec<_>>();
        let authorization = state
            .authorize_task(0, &producer, expected.clone())
            .unwrap();

        assert_eq!(authorization.physical_files, vec![ShuffleFileKey::sort(0)]);
        let report = ShuffleTaskPublication::new(expected.clone(), vec![], vec![]);
        assert_eq!(
            state.publish(&authorization, report.clone()).unwrap(),
            ShufflePublicationResult::Committed { event_cursor: None }
        );
        assert_eq!(state.sequence(), ShuffleExchangeSequence::INITIAL);
        assert!(!state.has_committed_artifacts());

        assert_eq!(
            state.publish(&authorization, report.clone()).unwrap(),
            ShufflePublicationResult::Replay { event_cursor: None }
        );

        let mut moved = producer.clone();
        moved.id = "executor-moved".into();
        let moved_authorization = state
            .authorize_task(0, &moved, expected)
            .unwrap();
        assert_eq!(
            state.publish(&moved_authorization, report).unwrap_err(),
            ShuffleExchangeError::ConflictingTaskReplay {
                producer_task_id: 0
            }
        );
    }

    #[test]
    fn sort_publication_requires_complete_report_manifest() {
        let mut state = exchange();
        let producer = executor(0);
        let expected = (0..OUTPUT_PARTITIONS).collect::<Vec<_>>();
        let authorization = state
            .authorize_task(0, &producer, expected.clone())
            .unwrap();
        let incomplete = (0..OUTPUT_PARTITIONS - 1).collect::<Vec<_>>();

        assert_eq!(
            state
                .publish(
                    &authorization,
                    ShuffleTaskPublication::new(
                        incomplete.clone(),
                        vec![0],
                        vec![location(0, 0, Some(0))],
                    ),
                )
                .unwrap_err(),
            ShuffleExchangeError::WrongTaskOutputPartitions {
                expected,
                actual: incomplete,
            }
        );
        assert_eq!(state.sequence(), ShuffleExchangeSequence::INITIAL);
        assert!(!state.is_task_accepted(0));
        assert!(!state.has_committed_artifacts());
    }

    #[test]
    fn sort_publication_rejects_missing_reader_visible_location() {
        let mut state = exchange();
        let producer = executor(0);
        let expected = (0..OUTPUT_PARTITIONS).collect::<Vec<_>>();
        let authorization = state
            .authorize_task(0, &producer, expected.clone())
            .unwrap();

        assert_eq!(
            state
                .publish(
                    &authorization,
                    ShuffleTaskPublication::new(
                        expected,
                        vec![0, 1],
                        vec![location(0, 0, Some(0))],
                    ),
                )
                .unwrap_err(),
            ShuffleExchangeError::WrongTaskOutputPartitions {
                expected: vec![0, 1],
                actual: vec![0],
            }
        );
        assert!(!state.is_task_accepted(0));
        assert!(!state.has_committed_artifacts());
    }

    #[test]
    fn authorization_derives_exact_physical_file_claims() {
        let state = passthrough_exchange();
        let authorization = state
            .authorize_task(2, &executor(2), vec![3, 1, 1])
            .unwrap();

        assert_eq!(authorization.expected_output_partitions, vec![1, 3]);
        assert_eq!(
            authorization.physical_files,
            vec![
                ShuffleFileKey::passthrough(1, Some(2)),
                ShuffleFileKey::passthrough(3, Some(2)),
            ]
        );
    }

    #[test]
    fn foreign_generation_rejects_empty_publication() {
        let mut state = exchange();
        let foreign_id = ShuffleExchangeId::new(JobId::from("other-job"), 7);
        let foreign = ShuffleExchangeState::new(
            foreign_id.clone(),
            OUTPUT_PARTITIONS,
            ShuffleLayout::Sort,
        );
        let expected = (0..OUTPUT_PARTITIONS).collect::<Vec<_>>();
        let authorization = foreign
            .authorize_task(0, &executor(0), expected.clone())
            .unwrap();

        assert_eq!(
            state
                .publish(
                    &authorization,
                    ShuffleTaskPublication::new(expected, vec![], vec![]),
                )
                .unwrap_err(),
            ShuffleExchangeError::WrongExchange {
                expected: exchange_id(),
                actual: foreign_id,
            }
        );
        assert!(!state.is_task_accepted(0));
    }

    #[test]
    fn conflicting_duplicates_fail_without_partial_publication() {
        let mut state = exchange();
        let epoch = state.epoch();
        let first = location(0, 0, Some(10));
        let mut conflict = first.clone();
        conflict.partition_stats = PartitionStats::new(Some(11), Some(1), Some(100));

        assert_eq!(
            publish(&mut state, epoch, 0, vec![first, conflict]).unwrap_err(),
            ShuffleExchangeError::ConflictingArtifact {
                key: ShuffleArtifactKey {
                    producer_task_id: 0,
                    output_partition_id: 0,
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

        publish(&mut state, epoch, 0, vec![artifact.clone(), artifact.clone()])
            .unwrap();

        let visible = state
            .artifacts_after(&partition_cursor(epoch, 0, 0))
            .unwrap();
        assert_eq!(visible.artifacts().len(), 1);
        assert!(same_artifact_location(
            visible.artifacts()[0].location(),
            &artifact
        ));
    }

    #[test]
    fn mixed_executor_publication_fails_atomically() {
        let mut state = exchange();
        let epoch = state.epoch();

        let first = location(0, 0, Some(10));
        let mut second = location(0, 1, Some(10));
        second.executor_meta.host = "other-host".into();

        assert_eq!(
            publish(&mut state, epoch, 0, vec![first, second]).unwrap_err(),
            ShuffleExchangeError::MixedExecutorPublication {
                producer_task_id: 0
            }
        );
        assert_eq!(state.sequence(), ShuffleExchangeSequence::INITIAL);
        assert!(!state.is_task_accepted(0));
        assert!(!state.has_committed_artifacts());
    }

    #[test]
    fn malformed_identity_fails_atomically() {
        let mut state = exchange();
        let epoch = state.epoch();

        let valid = location(0, 0, Some(9));
        let mut wrong_exchange = location(0, 1, Some(10));
        wrong_exchange.partition_id.stage_id = 8;
        assert!(matches!(
            publish(&mut state, epoch, 0, vec![valid, wrong_exchange]),
            Err(ShuffleExchangeError::WrongExchange { .. })
        ));

        let wrong_task = location(1, 0, Some(11));
        assert_eq!(
            publish(&mut state, epoch, 0, vec![wrong_task]).unwrap_err(),
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

        publish(&mut state, epoch, 0, vec![location(0, 0, Some(10))])
            .unwrap();
        publish(
            &mut state,
            epoch,
            1,
            vec![location(1, 0, Some(20)), location(1, 1, Some(21))],
        )
            .unwrap();

        let delta = state
            .artifacts_after(&partition_cursor(epoch, 0, 1))
            .unwrap();
        assert_eq!(delta.through_cursor(), &partition_cursor(epoch, 0, 2));
        assert_eq!(delta.artifacts().len(), 1);
        assert_eq!(delta.artifacts()[0].sequence().get(), 2);
        assert_eq!(delta.artifacts()[0].location().map_partition_id, 1);

        let caught_up = state
            .artifacts_after(&partition_cursor(epoch, 1, 2))
            .unwrap();
        assert!(caught_up.artifacts().is_empty());
        assert_eq!(caught_up.through_cursor(), &partition_cursor(epoch, 1, 2));

        assert_eq!(
            state
                .artifacts_after(&partition_cursor(epoch, 0, 3))
                .unwrap_err(),
            ShuffleExchangeError::CursorAhead {
                after: ShuffleExchangeSequence::new(3),
                current: ShuffleExchangeSequence::new(2),
            }
        );
    }

    #[test]
    fn partition_delta_is_detached_from_later_state_mutation() {
        let mut state = exchange();
        let epoch = state.epoch();

        publish(&mut state, epoch, 0, vec![location(0, 0, Some(10))]).unwrap();
        let snapshot = state
            .artifacts_after(&partition_cursor(epoch, 0, 0))
            .unwrap();

        // An owned snapshot must remain usable after the exchange mutates.
        publish(&mut state, epoch, 1, vec![location(1, 0, Some(20))]).unwrap();

        assert_eq!(
            snapshot.through_cursor(),
            &partition_cursor(epoch, 0, 1)
        );
        assert_eq!(snapshot.artifacts().len(), 1);
        assert_eq!(snapshot.artifacts()[0].location().file_id, Some(0));
        assert_eq!(state.sequence().get(), 2);
    }

    #[test]
    fn wrong_producer_file_id_fails_before_mutation() {
        let mut state = exchange();
        let mut malformed = location(1, 1, Some(1));
        malformed.file_id = Some(0);
        let producer = malformed.executor_meta.clone();
        let expected = (0..OUTPUT_PARTITIONS).collect::<Vec<_>>();
        let authorization = state
            .authorize_task(1, &producer, expected.clone())
            .unwrap();

        assert_eq!(
            state
                .publish(
                    &authorization,
                    ShuffleTaskPublication::new(expected, vec![1], vec![malformed]),
                )
                .unwrap_err(),
            ShuffleExchangeError::WrongProducerFileId {
                producer_task_id: 1,
                expected: 1,
                actual: Some(0),
            }
        );
        assert_eq!(state.sequence(), ShuffleExchangeSequence::INITIAL);
        assert!(!state.is_task_accepted(1));
    }

    #[test]
    fn preexisting_physical_owner_blocks_valid_publication() {
        let mut state = exchange();
        let epoch = state.epoch();

        // Models a retained path claim from lifetime fencing/recovery state.
        state.file_owners.insert(ShuffleFileKey::sort(0), 99);

        assert_eq!(
            publish(&mut state, epoch, 0, vec![location(0, 0, Some(0))])
                .unwrap_err(),
            ShuffleExchangeError::ConflictingFileOwnership {
                file: ShuffleFileKey::sort(0),
                existing_producer_task_id: 99,
                incoming_producer_task_id: 0,
            }
        );
        assert!(!state.is_task_accepted(0));
        assert!(!state.has_committed_artifacts());
        assert_eq!(state.sequence(), ShuffleExchangeSequence::INITIAL);
    }

    #[test]
    fn passthrough_files_are_bound_to_producer_task_id() {
        let mut state = passthrough_exchange();
        let epoch = state.epoch();

        publish(&mut state, epoch, 0, vec![passthrough_location(0, 0, Some(0))])
            .unwrap();

        let mut malformed = passthrough_location(1, 0, Some(1));
        malformed.file_id = Some(0);
        let producer = malformed.executor_meta.clone();
        let authorization = state
            .authorize_task(1, &producer, vec![0])
            .unwrap();
        assert_eq!(
            state
                .publish(
                    &authorization,
                    ShuffleTaskPublication::new(vec![0], vec![0], vec![malformed]),
                )
                .unwrap_err(),
            ShuffleExchangeError::WrongProducerFileId {
                producer_task_id: 1,
                expected: 1,
                actual: Some(0),
            }
        );

        assert!(matches!(
            publish(&mut state, epoch, 1, vec![passthrough_location(1, 0, Some(1))]),
            Ok(ShufflePublicationResult::Committed { .. })
        ));
    }

    #[test]
    fn retained_physical_owner_blocks_same_numeric_task_id() {
        let mut state = exchange();
        let epoch = state.epoch();

        // A rollback can recycle task id 0 while the old path claim must remain
        // fenced. Exact replay is impossible here because accepted_tasks is empty.
        state.file_owners.insert(ShuffleFileKey::sort(0), 0);

        assert_eq!(
            publish(&mut state, epoch, 0, vec![location(0, 0, Some(0))])
                .unwrap_err(),
            ShuffleExchangeError::ConflictingFileOwnership {
                file: ShuffleFileKey::sort(0),
                existing_producer_task_id: 0,
                incoming_producer_task_id: 0,
            }
        );
        assert!(!state.is_task_accepted(0));
        assert!(!state.has_committed_artifacts());
    }

    #[test]
    fn sort_authorization_requires_full_exchange_shape() {
        let state = exchange();

        assert_eq!(
            state
                .authorize_task(0, &executor(0), vec![0])
                .unwrap_err(),
            ShuffleExchangeError::WrongTaskOutputPartitions {
                expected: (0..OUTPUT_PARTITIONS).collect(),
                actual: vec![0],
            }
        );
    }

    #[test]
    fn passthrough_publication_must_cover_every_authorized_output() {
        let mut state = passthrough_exchange();
        let producer = executor(0);
        let location_zero = passthrough_location(0, 0, Some(0));
        let authorization = state
            .authorize_task(0, &producer, vec![0, 1])
            .unwrap();

        assert_eq!(
            state
                .publish(
                    &authorization,
                    ShuffleTaskPublication::new(vec![0, 1], vec![0, 1], vec![location_zero]),
                )
                .unwrap_err(),
            ShuffleExchangeError::WrongTaskOutputPartitions {
                expected: vec![0, 1],
                actual: vec![0],
            }
        );
        assert_eq!(state.sequence(), ShuffleExchangeSequence::INITIAL);
        assert!(!state.is_task_accepted(0));
        assert!(!state.has_committed_artifacts());
    }

    #[test]
    fn passthrough_zero_row_output_still_satisfies_completeness() {
        let mut state = passthrough_exchange();
        let producer = executor(0);
        let mut zero = passthrough_location(0, 1, Some(0));
        zero.partition_stats = PartitionStats::new(Some(0), Some(0), Some(0));
        let authorization = state
            .authorize_task(0, &producer, vec![0, 1])
            .unwrap();

        assert!(matches!(
            state.publish(
                &authorization,
                ShuffleTaskPublication::new(
                    vec![0, 1],
                    vec![0, 1],
                    vec![passthrough_location(0, 0, Some(0)), zero],
                ),
            ),
            Ok(ShufflePublicationResult::Committed { .. })
        ));
    }

    #[test]
    fn partition_delta_cursor_cannot_advance_a_sibling_partition() {
        let mut state = exchange();
        let epoch = state.epoch();

        publish(
            &mut state,
            epoch,
            0,
            vec![location(0, 0, Some(0)), location(0, 1, Some(0))],
        )
        .unwrap();

        let partition_zero = state
            .artifacts_after(&partition_cursor(epoch, 0, 0))
            .unwrap();
        assert_eq!(partition_zero.through_cursor().output_partition_id(), 0);
        assert_eq!(partition_zero.through_cursor().sequence().get(), 1);

        let partition_one = state
            .artifacts_after(&partition_cursor(epoch, 1, 0))
            .unwrap();
        assert_eq!(partition_one.artifacts().len(), 1);
        assert_eq!(partition_one.through_cursor().output_partition_id(), 1);
        assert_eq!(partition_one.through_cursor().sequence().get(), 1);
    }

    #[test]
    fn physical_file_key_matches_cross_layout_path_aliases() {
        assert_eq!(
            ShuffleFileKey::passthrough(10, None),
            ShuffleFileKey::sort(10)
        );
        assert_ne!(
            ShuffleFileKey::passthrough(10, Some(7)),
            ShuffleFileKey::sort(10)
        );
    }

    #[test]
    fn exchange_rejects_report_layout_different_from_planned_layout() {
        let mut state = exchange();
        let epoch = state.epoch();

        assert_eq!(state.layout(), ShuffleLayout::Sort);
        assert_eq!(
            publish(&mut state, epoch, 0, vec![passthrough_location(0, 1, Some(11))])
                .unwrap_err(),
            ShuffleExchangeError::WrongShuffleLayout {
                expected: ShuffleLayout::Sort,
                actual: ShuffleLayout::Passthrough,
            }
        );
        assert_eq!(state.sequence(), ShuffleExchangeSequence::INITIAL);
        assert!(!state.is_task_accepted(0));
    }

    #[test]
    fn one_sort_task_can_publish_multiple_partitions_from_one_file() {
        let mut state = exchange();
        let epoch = state.epoch();

        assert!(matches!(
            publish(
                &mut state,
                epoch,
                0,
                vec![location(0, 0, Some(10)), location(0, 1, Some(10))],
            ),
            Ok(ShufflePublicationResult::Committed { .. })
        ));
        assert_eq!(state.sequence().get(), 1);
        assert_eq!(
            state
                .artifacts_after(&partition_cursor(epoch, 0, 0))
                .unwrap()
                .artifacts()
                .len(),
            1
        );
        assert_eq!(
            state
                .artifacts_after(&partition_cursor(epoch, 1, 0))
                .unwrap()
                .artifacts()
                .len(),
            1
        );
    }

    #[test]
    fn distinct_file_id_on_same_output_partition_is_not_a_collision() {
        let mut state = exchange();
        let epoch = state.epoch();

        publish(&mut state, epoch, 0, vec![location(0, 0, Some(10))])
            .unwrap();

        assert!(matches!(
            publish(&mut state, epoch, 1, vec![location(1, 0, Some(11))]),
            Ok(ShufflePublicationResult::Committed { .. })
        ));
        assert_eq!(state.sequence().get(), 2);
    }

    #[test]
    fn partition_delta_advances_safely_through_other_partitions_and_seal() {
        let mut state = exchange();
        let epoch = state.epoch();

        publish(&mut state, epoch, 0, vec![location(0, 0, Some(10))])
            .unwrap();
        publish(&mut state, epoch, 1, vec![location(1, 1, Some(20))])
            .unwrap();

        let partition_zero = state
            .artifacts_after(&partition_cursor(epoch, 0, 0))
            .unwrap();
        assert_eq!(partition_zero.artifacts().len(), 1);
        assert_eq!(
            partition_zero.through_cursor(),
            &partition_cursor(epoch, 0, 2)
        );
        assert_eq!(partition_zero.lifecycle(), ShuffleExchangeLifecycle::Open);

        let partition_one = state
            .artifacts_after(&partition_cursor(epoch, 1, 0))
            .unwrap();
        assert_eq!(partition_one.artifacts().len(), 1);
        assert_eq!(
            partition_one.through_cursor(),
            &partition_cursor(epoch, 1, 2)
        );

        state.seal(epoch).unwrap();
        let sealed = state
            .artifacts_after(&partition_cursor(epoch, 0, 2))
            .unwrap();
        assert!(sealed.artifacts().is_empty());
        assert_eq!(sealed.through_cursor(), &partition_cursor(epoch, 0, 3));
        assert_eq!(sealed.lifecycle(), ShuffleExchangeLifecycle::Sealed);
    }

    #[test]
    fn sort_shuffle_without_file_id_fails_atomically() {
        let mut state = exchange();
        let epoch = state.epoch();

        assert_eq!(
            publish(&mut state, epoch, 0, vec![location(0, 0, None)]).unwrap_err(),
            ShuffleExchangeError::MissingSortShuffleFileId {
                producer_task_id: 0
            }
        );
        assert_eq!(state.sequence(), ShuffleExchangeSequence::INITIAL);
        assert!(!state.is_task_accepted(0));
        assert!(!state.has_committed_artifacts());
    }

    #[test]
    fn out_of_range_output_partition_fails_atomically() {
        let mut state = exchange();
        let epoch = state.epoch();

        assert_eq!(
            publish(
                &mut state,
                epoch,
                0,
                vec![location(0, OUTPUT_PARTITIONS, Some(10))],
            )
            .unwrap_err(),
            ShuffleExchangeError::InvalidOutputPartition {
                actual: OUTPUT_PARTITIONS,
                partition_count: OUTPUT_PARTITIONS,
            }
        );
        assert_eq!(state.sequence(), ShuffleExchangeSequence::INITIAL);
        assert!(!state.is_task_accepted(0));
        assert!(!state.has_committed_artifacts());

        assert!(matches!(
            state.artifacts_after(&partition_cursor(epoch, OUTPUT_PARTITIONS, 0)),
            Err(ShuffleExchangeError::InvalidOutputPartition { .. })
        ));
    }

    #[test]
    fn stale_epoch_is_rejected_without_mutation() {
        let mut state = exchange();
        let stale = ShuffleExchangeEpoch::INITIAL;
        let current = stale.checked_next().unwrap();
        state.generation = ShuffleExchangeGeneration::new(exchange_id(), current);

        assert_eq!(
            publish(
                &mut state,
                stale,
                0,
                vec![location(0, 0, Some(10))],
            )
            .unwrap_err(),
            ShuffleExchangeError::StaleEpoch {
                requested: stale,
                current,
            }
        );
        assert_eq!(state.sequence(), ShuffleExchangeSequence::INITIAL);
        assert!(!state.is_task_accepted(0));

        assert_eq!(
            state
                .artifacts_after(&partition_cursor(stale, 0, 0))
                .unwrap_err(),
            ShuffleExchangeError::StaleEpoch {
                requested: stale,
                current,
            }
        );
    }

    #[test]
    fn future_epoch_is_rejected_without_mutation() {
        let mut state = exchange();
        let current = state.epoch();
        let future = current.checked_next().unwrap();

        assert_eq!(
            publish(
                &mut state,
                future,
                0,
                vec![location(0, 0, Some(10))],
            )
            .unwrap_err(),
            ShuffleExchangeError::FutureEpoch {
                requested: future,
                current,
            }
        );
        assert_eq!(state.sequence(), ShuffleExchangeSequence::INITIAL);
        assert!(!state.is_task_accepted(0));

        assert_eq!(
            state
                .artifacts_after(&partition_cursor(future, 0, 0))
                .unwrap_err(),
            ShuffleExchangeError::FutureEpoch {
                requested: future,
                current,
            }
        );
    }

    #[test]
    fn cursor_from_another_exchange_is_rejected() {
        let state = exchange();
        let foreign = ShufflePartitionCursor::new(
            cursor_for(
                ShuffleExchangeId::new(JobId::from("other-job"), 7),
                state.epoch(),
                0,
            ),
            0,
        );

        assert_eq!(
            state.artifacts_after(&foreign).unwrap_err(),
            ShuffleExchangeError::WrongExchange {
                expected: exchange_id(),
                actual: ShuffleExchangeId::new(JobId::from("other-job"), 7),
            }
        );
    }

    #[test]
    fn sequence_exhaustion_reserves_the_final_position_for_seal() {
        let mut state = exchange();
        state.sequence = ShuffleExchangeSequence::new(u64::MAX - 1);
        let epoch = state.epoch();

        // A non-empty publication would consume the last sequence and make
        // the open exchange impossible to seal, so it fails closed.
        assert_eq!(
            publish(&mut state, epoch, 0, vec![location(0, 0, Some(0))])
                .unwrap_err(),
            ShuffleExchangeError::SequenceExhausted
        );
        assert!(!state.is_task_accepted(0));
        assert!(!state.has_committed_artifacts());

        // Empty output consumes no sequence and remains safe while one seal
        // sequence is still available.
        assert_eq!(
            publish(&mut state, epoch, 1, vec![]).unwrap(),
            ShufflePublicationResult::Committed { event_cursor: None }
        );
        assert!(state.is_task_accepted(1));
        assert_eq!(state.sequence().get(), u64::MAX - 1);

        assert_eq!(
            state.seal(epoch).unwrap().map(|cursor| cursor.sequence().get()),
            Some(u64::MAX)
        );
        assert_eq!(state.lifecycle(), ShuffleExchangeLifecycle::Sealed);

        // An artificially open state already at MAX accepts no unseen
        // mutation and cannot fabricate a wrapped seal event.
        let mut exhausted = exchange();
        exhausted.sequence = ShuffleExchangeSequence::new(u64::MAX);
        let exhausted_epoch = exhausted.epoch();
        assert_eq!(
            publish(&mut exhausted, exhausted_epoch, 0, vec![]).unwrap_err(),
            ShuffleExchangeError::SequenceExhausted
        );
        assert_eq!(
            exhausted.seal(exhausted_epoch).unwrap_err(),
            ShuffleExchangeError::SequenceExhausted
        );
        assert!(!exhausted.is_task_accepted(0));
        assert_eq!(exhausted.lifecycle(), ShuffleExchangeLifecycle::Open);
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
