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

use std::collections::{BTreeSet, HashMap, HashSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

use ballista_core::serde::scheduler::{PartitionLocation, ShuffleLayout};
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
    file_id: Option<u64>,
}

impl ShuffleArtifactKey {
    /// Creates the canonical producer-ownership key for one artifact.
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

/// Exchange-local identity of one physical shuffle data-file path.
///
/// This models the relative path below job/stage exactly. Both sort file 10 and
/// passthrough partition 10 without a file id resolve to 10/data.arrow and must
/// therefore compare equal even though their layouts differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ShuffleFileKey {
    directory_id: u128,
    file_name: ShuffleFileName,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
    fn from_location(location: &PartitionLocation) -> Self {
        Self {
            executor_id: location.executor_meta.id.clone(),
            host: location.executor_meta.host.clone(),
            port: location.executor_meta.port,
        }
    }
}

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
    Replay,
}

/// Description of one successful exchange-epoch rollover.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ShuffleEpochRollover {
    /// Epoch that was retired.
    pub(crate) previous_epoch: ShuffleExchangeEpoch,
    /// Fresh epoch containing only surviving producer output.
    pub(crate) current_epoch: ShuffleExchangeEpoch,
    /// Bootstrap cursor exposing every retained artifact in the fresh epoch.
    ///
    /// This is None when no materialized artifacts survive. Accepted empty
    /// publications may still survive without creating a reader-visible event.
    pub(crate) bootstrap_cursor: Option<ShuffleExchangeCursor>,
    /// Accepted producer tasks removed by this rollover, in deterministic order.
    pub(crate) invalidated_tasks: Vec<usize>,
    /// Number of accepted producer tasks retained in the fresh epoch.
    pub(crate) retained_task_count: usize,
    /// Number of materialized artifacts retained in the fresh epoch.
    pub(crate) retained_artifact_count: usize,
}

/// Result of applying an output-loss signal to an exchange.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ShuffleInvalidationResult {
    /// No accepted producer output matched the specific loss signal.
    Unchanged,
    /// The old materialized generation was retired.
    Rolled(ShuffleEpochRollover),
}

/// Atomic snapshot of one output partition through a global exchange cursor.
///
/// A consumer may persist through_cursor only after consuming every artifact
/// returned in this delta. That makes the global cursor safe for this partition
/// even when unrelated partitions advanced the exchange sequence.
#[derive(Debug)]
pub(crate) struct ShufflePartitionDelta<'a> {
    through_cursor: ShuffleExchangeCursor,
    lifecycle: ShuffleExchangeLifecycle,
    artifacts: Vec<&'a PublishedShuffleArtifact>,
}

impl<'a> ShufflePartitionDelta<'a> {
    /// Returns the cursor through which this output partition was scanned.
    pub(crate) fn through_cursor(&self) -> &ShuffleExchangeCursor {
        &self.through_cursor
    }

    /// Returns the exchange lifecycle observed with this delta snapshot.
    pub(crate) fn lifecycle(&self) -> ShuffleExchangeLifecycle {
        self.lifecycle
    }

    /// Returns artifacts published after the caller's cursor through this snapshot.
    pub(crate) fn artifacts(&self) -> &[&'a PublishedShuffleArtifact] {
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
    /// An operation targeted an epoch that is no longer current.
    StaleEpoch {
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
    /// A producer attempted to reuse a physical file identity retired by recovery.
    RetiredFileReuse {
        file: ShuffleFileKey,
        producer_task_id: usize,
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
    /// A task replay did not exactly match the task publication already accepted.
    ConflictingTaskReplay { producer_task_id: usize },
    /// A previously unseen task attempted to publish after the epoch sealed.
    PublicationAfterSeal,
    /// A reader cursor is ahead of the exchange's current event sequence.
    CursorAhead {
        after: ShuffleExchangeSequence,
        current: ShuffleExchangeSequence,
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
            Self::RetiredFileReuse {
                file,
                producer_task_id,
            } => write!(
                f,
                "shuffle file {file:?} was retired by recovery and cannot be reused by producer task {producer_task_id}"
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
/// Epoch rollover preserves usable output while fencing readers pinned to the
/// retired generation. Physical file identities removed by recovery remain
/// retired for the lifetime of this logical exchange because the current
/// on-disk shuffle path does not encode the exchange epoch.
#[derive(Debug)]
pub(crate) struct ShuffleExchangeState {
    generation: ShuffleExchangeGeneration,
    output_partition_count: usize,
    layout: ShuffleLayout,
    sequence: ShuffleExchangeSequence,
    lifecycle: ShuffleExchangeLifecycle,
    artifacts: HashMap<ShuffleArtifactKey, PublishedShuffleArtifact>,
    // Physical shuffle file -> owning producer task in the current epoch.
    file_owners: HashMap<ShuffleFileKey, usize>,
    // Physical file identities retired by any prior epoch. These stay fenced
    // because stale readers can still address the old path and the path itself
    // does not contain the exchange epoch.
    retired_files: HashSet<ShuffleFileKey>,
    // Output partition -> append-ordered (generation-local sequence, artifact key).
    // The state owns the generation, so duplicating a full cursor per entry would
    // repeat the same job/stage identity for every artifact.
    partition_index:
        HashMap<usize, Vec<(ShuffleExchangeSequence, ShuffleArtifactKey)>>,
    // Producer task -> canonical artifact keys. Empty output is represented by
    // an empty vector, which is still an accepted task publication.
    //
    // Producer task IDs identify accepted task attempts within this epoch.
    // Rolling the epoch fences logical history, but it does not by itself make
    // reuse of an old physical file path safe because the current shuffle path
    // does not encode the epoch. The recovery follow-up must preserve that
    // separate physical-file lifetime invariant.
    accepted_tasks: HashMap<usize, Vec<ShuffleArtifactKey>>,
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
            artifacts: HashMap::new(),
            file_owners: HashMap::new(),
            retired_files: HashSet::new(),
            partition_index: HashMap::new(),
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

    /// Returns the current generation-bound reader position.
    pub(crate) fn cursor(&self) -> ShuffleExchangeCursor {
        self.cursor_at(self.sequence)
    }

    /// Returns the latest reader-visible event sequence.
    ///
    /// This is primarily useful for diagnostics and tests. Reader APIs use
    /// [ShuffleExchangeCursor] so an event position cannot be detached from
    /// the generation whose history it indexes.
    pub(crate) fn sequence(&self) -> ShuffleExchangeSequence {
        self.sequence
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

    /// Returns whether a physical shuffle path is current or permanently retired.
    ///
    /// Scheduler integration must use this before dispatching a writer. Rejecting
    /// a publication after the executor has already overwritten a retired path is
    /// only a last-line consistency check and cannot protect stale in-flight reads.
    pub(crate) fn is_file_reserved(&self, file: ShuffleFileKey) -> bool {
        self.file_owners.contains_key(&file) || self.retired_files.contains(&file)
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

            let file = ShuffleFileKey::from_location(location)
                .expect("canonical publication must have a valid shuffle file identity");
            if self.retired_files.contains(&file) {
                return Err(ShuffleExchangeError::RetiredFileReuse {
                    file,
                    producer_task_id,
                });
            }
            match self.file_owners.get(&file) {
                Some(existing_producer_task_id)
                    if *existing_producer_task_id != producer_task_id =>
                {
                    return Err(ShuffleExchangeError::ConflictingFileOwnership {
                        file,
                        existing_producer_task_id: *existing_producer_task_id,
                        incoming_producer_task_id: producer_task_id,
                    });
                }
                _ => {}
            }
        }

        if publication.is_empty() {
            self.accepted_tasks.insert(producer_task_id, Vec::new());
            return Ok(ShufflePublicationResult::Committed {
                event_cursor: None,
            });
        }

        let next_sequence = self
            .sequence
            .checked_next()
            .ok_or(ShuffleExchangeError::SequenceExhausted)?;
        let event_cursor = self.cursor_at(next_sequence);
        let mut keys = Vec::with_capacity(publication.len());

        for location in publication {
            let key = ShuffleArtifactKey::from_location(&location);
            let file = ShuffleFileKey::from_location(&location)
                .expect("canonical publication must have a valid shuffle file identity");
            self.file_owners.insert(file, producer_task_id);
            self.artifacts.insert(
                key,
                PublishedShuffleArtifact {
                    sequence: next_sequence,
                    location,
                },
            );
            self.partition_index
                .entry(key.output_partition_id)
                .or_default()
                .push((next_sequence, key));
            keys.push(key);
        }

        self.accepted_tasks.insert(producer_task_id, keys);
        self.sequence = next_sequence;
        Ok(ShufflePublicationResult::Committed {
            event_cursor: Some(event_cursor),
        })
    }

    /// Invalidates accepted output for the specified producer tasks.
    ///
    /// Recovery is task-granular: losing one output from a task retires that
    /// task's whole accepted publication. Unknown task ids are ignored so
    /// duplicate or delayed loss signals cannot churn the epoch.
    pub(crate) fn invalidate_tasks(
        &mut self,
        epoch: ShuffleExchangeEpoch,
        lost_tasks: &[usize],
    ) -> Result<ShuffleInvalidationResult> {
        self.ensure_epoch(epoch)?;

        let invalidated: BTreeSet<_> = lost_tasks
            .iter()
            .copied()
            .filter(|task| self.accepted_tasks.contains_key(task))
            .collect();
        self.roll_epoch(invalidated, false)
    }

    /// Invalidates the producer tasks owning the specified materialized artifacts.
    ///
    /// Artifact loss is promoted to task loss because Ballista retries producer
    /// output at task granularity. Unknown artifact keys are harmless no-ops.
    pub(crate) fn invalidate_artifacts(
        &mut self,
        epoch: ShuffleExchangeEpoch,
        lost_artifacts: &[ShuffleArtifactKey],
    ) -> Result<ShuffleInvalidationResult> {
        self.ensure_epoch(epoch)?;

        let invalidated: BTreeSet<_> = lost_artifacts
            .iter()
            .filter(|key| self.artifacts.contains_key(key))
            .map(|key| key.producer_task_id())
            .collect();
        self.roll_epoch(invalidated, false)
    }

    /// Retires the entire current producer generation.
    ///
    /// A full producer-attempt reset always rolls the epoch, even when no task
    /// has published output yet. Subscribers may already hold a cursor for the
    /// open empty generation and must not silently cross the attempt boundary.
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

        let next_sequence = self
            .sequence
            .checked_next()
            .ok_or(ShuffleExchangeError::SequenceExhausted)?;
        self.lifecycle = ShuffleExchangeLifecycle::Sealed;
        self.sequence = next_sequence;
        Ok(Some(self.cursor_at(next_sequence)))
    }

    /// Returns an atomic delta snapshot for one output partition.
    ///
    /// The returned artifacts are ordered by publication sequence and then by
    /// canonical artifact key within a task-atomic publication. through_cursor
    /// is the exact global cursor through which this partition has been scanned.
    /// Lifecycle is captured in the same snapshot so a seal remains observable
    /// even when no new artifact exists for this partition.
    pub(crate) fn artifacts_after(
        &self,
        after: &ShuffleExchangeCursor,
        output_partition_id: usize,
    ) -> Result<ShufflePartitionDelta<'_>> {
        self.ensure_cursor_generation(after)?;
        self.ensure_output_partition(output_partition_id)?;
        if after.sequence() > self.sequence {
            return Err(ShuffleExchangeError::CursorAhead {
                after: after.sequence(),
                current: self.sequence,
            });
        }

        let artifacts = if let Some(entries) = self.partition_index.get(&output_partition_id) {
            let first_new =
                entries.partition_point(|(sequence, _)| *sequence <= after.sequence());
            entries[first_new..]
                .iter()
                .map(|(_, key)| {
                    self.artifacts
                        .get(key)
                        .expect("shuffle partition index must reference a committed artifact")
                })
                .collect()
        } else {
            Vec::new()
        };

        Ok(ShufflePartitionDelta {
            through_cursor: self.cursor(),
            lifecycle: self.lifecycle,
            artifacts,
        })
    }

    fn roll_epoch(
        &mut self,
        invalidated_tasks: BTreeSet<usize>,
        force_rollover: bool,
    ) -> Result<ShuffleInvalidationResult> {
        if invalidated_tasks.is_empty() && !force_rollover {
            return Ok(ShuffleInvalidationResult::Unchanged);
        }

        // Allocate the fresh generation before constructing any replacement
        // state so exhaustion leaves the old generation completely untouched.
        let previous_epoch = self.epoch();
        let current_epoch = previous_epoch
            .checked_next()
            .ok_or(ShuffleExchangeError::EpochExhausted)?;
        let current_generation =
            ShuffleExchangeGeneration::new(self.id().clone(), current_epoch);

        let mut newly_retired = HashSet::new();
        let mut retained = Vec::with_capacity(self.artifacts.len());
        for (key, artifact) in &self.artifacts {
            let file = ShuffleFileKey::from_location(&artifact.location)
                .expect("committed shuffle artifact must have a valid file identity");
            if invalidated_tasks.contains(&key.producer_task_id) {
                newly_retired.insert(file);
            } else {
                retained.push((*key, artifact.location.clone()));
            }
        }
        retained.sort_by_key(|(key, _)| *key);

        let bootstrap_sequence = if retained.is_empty() {
            ShuffleExchangeSequence::INITIAL
        } else {
            ShuffleExchangeSequence::INITIAL
                .checked_next()
                .expect("initial shuffle sequence must advance")
        };
        let bootstrap_cursor = if retained.is_empty() {
            None
        } else {
            Some(ShuffleExchangeCursor::new(
                current_generation.clone(),
                bootstrap_sequence,
            ))
        };

        let mut artifacts = HashMap::with_capacity(retained.len());
        let mut file_owners = HashMap::new();
        let mut partition_index:
            HashMap<usize, Vec<(ShuffleExchangeSequence, ShuffleArtifactKey)>> =
            HashMap::new();

        for (key, location) in retained {
            let file = ShuffleFileKey::from_location(&location)
                .expect("retained shuffle artifact must have a valid file identity");
            match file_owners.insert(file, key.producer_task_id) {
                Some(existing) if existing != key.producer_task_id => {
                    panic!("retained shuffle file has conflicting producer owners")
                }
                _ => {}
            }
            artifacts.insert(
                key,
                PublishedShuffleArtifact {
                    sequence: bootstrap_sequence,
                    location,
                },
            );
            partition_index
                .entry(key.output_partition_id)
                .or_default()
                .push((bootstrap_sequence, key));
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

        self.generation = current_generation;
        self.sequence = bootstrap_sequence;
        self.lifecycle = ShuffleExchangeLifecycle::Open;
        self.artifacts = artifacts;
        self.file_owners = file_owners;
        self.retired_files.extend(newly_retired);
        self.partition_index = partition_index;
        self.accepted_tasks = accepted_tasks;

        Ok(ShuffleInvalidationResult::Rolled(rollover))
    }

    fn cursor_at(&self, sequence: ShuffleExchangeSequence) -> ShuffleExchangeCursor {
        ShuffleExchangeCursor::new(self.generation.clone(), sequence)
    }

    fn ensure_cursor_generation(&self, cursor: &ShuffleExchangeCursor) -> Result<()> {
        if cursor.exchange_id() != self.id() {
            return Err(ShuffleExchangeError::WrongExchange {
                expected: self.id().clone(),
                actual: cursor.exchange_id().clone(),
            });
        }
        self.ensure_epoch(cursor.epoch())
    }

    fn ensure_epoch(&self, epoch: ShuffleExchangeEpoch) -> Result<()> {
        let current = self.epoch();
        if epoch == current {
            Ok(())
        } else {
            Err(ShuffleExchangeError::StaleEpoch {
                requested: epoch,
                current,
            })
        }
    }

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

    fn canonical_publication(
        &self,
        producer_task_id: usize,
        locations: Vec<PartitionLocation>,
    ) -> Result<Vec<PartitionLocation>> {
        let mut by_key = HashMap::with_capacity(locations.len());
        let mut publication_endpoint: Option<ShuffleDataEndpoint> = None;

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

            let endpoint = ShuffleDataEndpoint::from_location(&location);
            match publication_endpoint.as_ref() {
                Some(expected) if expected != &endpoint => {
                    return Err(ShuffleExchangeError::MixedExecutorPublication {
                        producer_task_id,
                    });
                }
                None => publication_endpoint = Some(endpoint),
                _ => {}
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

    fn passthrough_location(
        task: usize,
        output_partition: usize,
        file_id: Option<u64>,
    ) -> PartitionLocation {
        let mut location = location(task, output_partition, file_id);
        location.is_sort_shuffle = false;
        location
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
            .artifacts_after(&cursor(epoch, 0), 0)
            .unwrap();
        assert_eq!(partition_zero.through_cursor(), &cursor(epoch, 1));
        assert_eq!(partition_zero.lifecycle(), ShuffleExchangeLifecycle::Open);
        assert_eq!(partition_zero.artifacts().len(), 2);
        assert!(partition_zero
            .artifacts()
            .iter()
            .all(|artifact| artifact.sequence().get() == 1));
        assert_eq!(partition_zero.artifacts()[0].location().file_id, Some(10));
        assert_eq!(partition_zero.artifacts()[1].location().file_id, Some(12));

        let partition_one = state
            .artifacts_after(&cursor(epoch, 0), 1)
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
        metadata_refresh[0].executor_meta.grpc_port += 1;
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
            .artifacts_after(&cursor(epoch, 0), 0)
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
            state.publish(epoch, 0, vec![first, second]).unwrap_err(),
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
            .artifacts_after(&cursor(epoch, 1), 0)
            .unwrap();
        assert_eq!(delta.through_cursor(), &cursor(epoch, 2));
        assert_eq!(delta.artifacts().len(), 1);
        assert_eq!(delta.artifacts()[0].sequence().get(), 2);
        assert_eq!(delta.artifacts()[0].location().map_partition_id, 1);

        let caught_up = state
            .artifacts_after(&cursor(epoch, 2), 1)
            .unwrap();
        assert!(caught_up.artifacts().is_empty());
        assert_eq!(caught_up.through_cursor(), &cursor(epoch, 2));

        assert_eq!(
            state
                .artifacts_after(&cursor(epoch, 3), 0)
                .unwrap_err(),
            ShuffleExchangeError::CursorAhead {
                after: ShuffleExchangeSequence::new(3),
                current: ShuffleExchangeSequence::new(2),
            }
        );
    }

    #[test]
    fn sort_file_ownership_collision_across_tasks_fails_before_mutation() {
        let mut state = exchange();
        let epoch = state.epoch();

        state
            .publish(epoch, 0, vec![location(0, 0, Some(10))])
            .unwrap();

        // Sort shuffle ignores output partition in its physical path, so a
        // different task cannot reuse file id 10 even for another partition.
        let alias = location(1, 1, Some(10));

        assert_eq!(
            state.publish(epoch, 1, vec![alias]).unwrap_err(),
            ShuffleExchangeError::ConflictingFileOwnership {
                file: ShuffleFileKey::sort(10),
                existing_producer_task_id: 0,
                incoming_producer_task_id: 1,
            }
        );
        assert_eq!(state.sequence().get(), 1);
        assert!(!state.is_task_accepted(1));
    }

    #[test]
    fn passthrough_file_ownership_matches_partitioned_path_shape() {
        let mut state = passthrough_exchange();
        let epoch = state.epoch();

        state
            .publish(epoch, 0, vec![passthrough_location(0, 0, Some(10))])
            .unwrap();

        assert_eq!(
            state
                .publish(epoch, 1, vec![passthrough_location(1, 0, Some(10))])
                .unwrap_err(),
            ShuffleExchangeError::ConflictingFileOwnership {
                file: ShuffleFileKey::passthrough(0, Some(10)),
                existing_producer_task_id: 0,
                incoming_producer_task_id: 1,
            }
        );

        // A different output partition is a different passthrough path.
        assert!(matches!(
            state.publish(epoch, 2, vec![passthrough_location(2, 1, Some(10))]),
            Ok(ShufflePublicationResult::Committed { .. })
        ));
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
            state
                .publish(epoch, 0, vec![passthrough_location(0, 1, Some(11))])
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
            state.publish(
                epoch,
                0,
                vec![location(0, 0, Some(10)), location(0, 1, Some(10))],
            ),
            Ok(ShufflePublicationResult::Committed { .. })
        ));
        assert_eq!(state.sequence().get(), 1);
        assert_eq!(
            state
                .artifacts_after(&cursor(epoch, 0), 0)
                .unwrap()
                .artifacts()
                .len(),
            1
        );
        assert_eq!(
            state
                .artifacts_after(&cursor(epoch, 0), 1)
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

        state
            .publish(epoch, 0, vec![location(0, 0, Some(10))])
            .unwrap();

        assert!(matches!(
            state.publish(epoch, 1, vec![location(1, 0, Some(11))]),
            Ok(ShufflePublicationResult::Committed { .. })
        ));
        assert_eq!(state.sequence().get(), 2);
    }

    #[test]
    fn partition_delta_advances_safely_through_other_partitions_and_seal() {
        let mut state = exchange();
        let epoch = state.epoch();

        state
            .publish(epoch, 0, vec![location(0, 0, Some(10))])
            .unwrap();
        state
            .publish(epoch, 1, vec![location(1, 1, Some(20))])
            .unwrap();

        let partition_zero = state
            .artifacts_after(&cursor(epoch, 0), 0)
            .unwrap();
        assert_eq!(partition_zero.artifacts().len(), 1);
        assert_eq!(partition_zero.through_cursor(), &cursor(epoch, 2));
        assert_eq!(partition_zero.lifecycle(), ShuffleExchangeLifecycle::Open);

        let partition_one = state
            .artifacts_after(&cursor(epoch, 0), 1)
            .unwrap();
        assert_eq!(partition_one.artifacts().len(), 1);
        assert_eq!(partition_one.through_cursor(), &cursor(epoch, 2));

        state.seal(epoch).unwrap();
        let sealed = state
            .artifacts_after(&cursor(epoch, 2), 0)
            .unwrap();
        assert!(sealed.artifacts().is_empty());
        assert_eq!(sealed.through_cursor(), &cursor(epoch, 3));
        assert_eq!(sealed.lifecycle(), ShuffleExchangeLifecycle::Sealed);
    }

    #[test]
    fn sort_shuffle_without_file_id_fails_atomically() {
        let mut state = exchange();
        let epoch = state.epoch();

        assert_eq!(
            state.publish(epoch, 0, vec![location(0, 0, None)]).unwrap_err(),
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
            state
                .publish(
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
            state.artifacts_after(&cursor(epoch, 0), OUTPUT_PARTITIONS),
            Err(ShuffleExchangeError::InvalidOutputPartition { .. })
        ));
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
                .artifacts_after(&cursor(stale, 0), 0)
                .unwrap_err(),
            ShuffleExchangeError::StaleEpoch {
                requested: stale,
                current: ShuffleExchangeEpoch::INITIAL,
            }
        );
    }

    #[test]
    fn cursor_from_another_exchange_is_rejected() {
        let state = exchange();
        let foreign = cursor_for(
            ShuffleExchangeId::new(JobId::from("other-job"), 7),
            state.epoch(),
            0,
        );

        assert_eq!(
            state.artifacts_after(&foreign, 0).unwrap_err(),
            ShuffleExchangeError::WrongExchange {
                expected: exchange_id(),
                actual: ShuffleExchangeId::new(JobId::from("other-job"), 7),
            }
        );
    }

    #[test]
    fn task_invalidation_rebases_survivors_and_retires_lost_files() {
        let mut state = exchange();
        let epoch_one = state.epoch();
        let task_zero = vec![location(0, 0, Some(10)), location(0, 1, Some(10))];
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
        assert_eq!(rollover.bootstrap_cursor, Some(cursor(epoch_two, 1)));
        assert_eq!(rollover.invalidated_tasks, vec![0]);
        assert_eq!(rollover.retained_task_count, 1);
        assert_eq!(rollover.retained_artifact_count, 1);
        assert_eq!(state.cursor(), cursor(epoch_two, 1));
        assert_eq!(state.lifecycle(), ShuffleExchangeLifecycle::Open);
        assert!(!state.is_task_accepted(0));
        assert!(state.is_task_accepted(1));
        assert!(state.is_file_reserved(ShuffleFileKey::sort(10)));
        assert!(state.is_file_reserved(ShuffleFileKey::sort(20)));

        assert!(matches!(
            state.artifacts_after(&cursor(epoch_one, 0), 0),
            Err(ShuffleExchangeError::StaleEpoch { .. })
        ));

        assert_eq!(
            state.publish(epoch_two, 1, task_one).unwrap(),
            ShufflePublicationResult::Replay
        );

        assert_eq!(
            state
                .publish(epoch_two, 2, vec![location(2, 2, Some(10))])
                .unwrap_err(),
            ShuffleExchangeError::RetiredFileReuse {
                file: ShuffleFileKey::sort(10),
                producer_task_id: 2,
            }
        );

        assert_eq!(
            state
                .publish(epoch_two, 2, vec![location(2, 0, Some(30))])
                .unwrap(),
            ShufflePublicationResult::Committed {
                event_cursor: Some(cursor(epoch_two, 2))
            }
        );

        assert!(matches!(
            state.publish(epoch_two, 3, vec![location(3, 1, Some(20))]),
            Err(ShuffleExchangeError::ConflictingFileOwnership { .. })
        ));
    }

    #[test]
    fn artifact_loss_invalidates_whole_producer_task() {
        let mut state = exchange();
        let epoch_one = state.epoch();
        let task_zero = vec![location(0, 0, Some(10)), location(0, 1, Some(10))];

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
        assert!(state.retired_files.contains(&ShuffleFileKey::sort(10)));
    }

    #[test]
    fn accepted_empty_survivor_is_retained_without_bootstrap_artifact() {
        let mut state = exchange();
        let epoch_one = state.epoch();

        state
            .publish(epoch_one, 0, vec![location(0, 0, Some(10))])
            .unwrap();
        state.publish(epoch_one, 1, vec![]).unwrap();

        let result = state.invalidate_tasks(epoch_one, &[0]).unwrap();
        let ShuffleInvalidationResult::Rolled(rollover) = result else {
            panic!("expected epoch rollover")
        };

        assert_eq!(rollover.bootstrap_cursor, None);
        assert_eq!(rollover.retained_task_count, 1);
        assert_eq!(rollover.retained_artifact_count, 0);
        assert_eq!(state.sequence(), ShuffleExchangeSequence::INITIAL);
        assert!(state.is_task_accepted(1));
        assert_eq!(
            state.publish(state.epoch(), 1, vec![]).unwrap(),
            ShufflePublicationResult::Replay
        );
    }

    #[test]
    fn invalidate_all_retires_materialized_files_and_empty_publications() {
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
        assert_eq!(state.cursor(), cursor(state.epoch(), 0));
        assert_eq!(state.lifecycle(), ShuffleExchangeLifecycle::Open);
        assert!(!state.has_committed_artifacts());
        assert!(!state.is_task_accepted(0));
        assert!(!state.is_task_accepted(1));
        assert!(!state.is_task_accepted(2));
        assert!(state.retired_files.contains(&ShuffleFileKey::sort(10)));
        assert!(state.retired_files.contains(&ShuffleFileKey::sort(20)));
    }

    #[test]
    fn invalidate_all_without_output_still_retires_generation() {
        let mut state = exchange();
        let epoch_one = state.epoch();

        let result = state.invalidate_all(epoch_one).unwrap();
        let ShuffleInvalidationResult::Rolled(rollover) = result else {
            panic!("expected forced epoch rollover")
        };

        assert_eq!(rollover.previous_epoch, epoch_one);
        assert_eq!(rollover.current_epoch, state.epoch());
        assert!(state.epoch() > epoch_one);
        assert!(rollover.invalidated_tasks.is_empty());
        assert_eq!(rollover.retained_task_count, 0);
        assert_eq!(rollover.retained_artifact_count, 0);
        assert_eq!(rollover.bootstrap_cursor, None);
        assert_eq!(state.sequence(), ShuffleExchangeSequence::INITIAL);
        assert!(state.retired_files.is_empty());
    }

    #[test]
    fn unknown_specific_loss_does_not_churn_epoch() {
        let mut state = exchange();
        let epoch = state.epoch();
        state
            .publish(epoch, 0, vec![location(0, 0, Some(10))])
            .unwrap();

        assert_eq!(
            state.invalidate_tasks(epoch, &[99, 99]).unwrap(),
            ShuffleInvalidationResult::Unchanged
        );
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
    fn invalidation_reopens_sealed_exchange_and_fences_old_cursor() {
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
            state.artifacts_after(&cursor(epoch_one, 0), 0),
            Err(ShuffleExchangeError::StaleEpoch { .. })
        ));
        assert!(matches!(
            state.invalidate_all(epoch_one),
            Err(ShuffleExchangeError::StaleEpoch { .. })
        ));
        assert_eq!(state.epoch(), epoch_two);
    }

    #[test]
    fn retired_file_fence_survives_multiple_epoch_rollovers() {
        let mut state = exchange();
        let epoch_one = state.epoch();
        state
            .publish(epoch_one, 0, vec![location(0, 0, Some(10))])
            .unwrap();
        state.invalidate_all(epoch_one).unwrap();

        let epoch_two = state.epoch();
        state
            .publish(epoch_two, 1, vec![location(1, 0, Some(20))])
            .unwrap();
        state.invalidate_all(epoch_two).unwrap();

        let epoch_three = state.epoch();
        assert!(matches!(
            state.publish(epoch_three, 2, vec![location(2, 0, Some(10))]),
            Err(ShuffleExchangeError::RetiredFileReuse { .. })
        ));
        assert!(matches!(
            state.publish(epoch_three, 3, vec![location(3, 0, Some(20))]),
            Err(ShuffleExchangeError::RetiredFileReuse { .. })
        ));
    }

    #[test]
    fn epoch_exhaustion_leaves_previous_state_intact() {
        let mut state = exchange();
        let epoch = ShuffleExchangeEpoch::new(u64::MAX).unwrap();
        state.generation = ShuffleExchangeGeneration::new(exchange_id(), epoch);
        state.sequence = ShuffleExchangeSequence::INITIAL;
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
        assert!(state.retired_files.is_empty());
    }

    #[test]
    fn sequence_exhaustion_fails_closed_without_mutation() {
        let mut state = exchange();
        state.sequence = ShuffleExchangeSequence::new(u64::MAX);
        let epoch = state.epoch();

        assert_eq!(
            state
                .publish(epoch, 0, vec![location(0, 0, Some(10))])
                .unwrap_err(),
            ShuffleExchangeError::SequenceExhausted
        );
        assert!(!state.is_task_accepted(0));
        assert!(!state.has_committed_artifacts());
        assert_eq!(state.sequence().get(), u64::MAX);
        assert_eq!(state.lifecycle(), ShuffleExchangeLifecycle::Open);

        assert_eq!(
            state.seal(epoch).unwrap_err(),
            ShuffleExchangeError::SequenceExhausted
        );
        assert_eq!(state.sequence().get(), u64::MAX);
        assert_eq!(state.lifecycle(), ShuffleExchangeLifecycle::Open);
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
