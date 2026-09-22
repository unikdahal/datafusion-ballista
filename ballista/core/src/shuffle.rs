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

//! Domain primitives for scheduler-owned shuffle exchanges.
//!
//! A shuffle exchange is the logical output of one producer stage. Its identity
//! is stable for the lifetime of the job; task or stage retries do not
//! manufacture a new logical exchange. When recovery invalidates previously
//! accepted materialized output, the exchange advances to a new
//! [ShuffleExchangeEpoch].
//!
//! Epochs are scoped to a [ShuffleExchangeId]. An epoch value has no meaning
//! independently of the exchange that owns it, and an epoch must never be reused
//! for the same exchange while a reader or writer from an older materialized
//! history may still exist. A recovery implementation must therefore preserve
//! the current epoch across scheduler reconstruction or invalidate all
//! pre-recovery exchange handles before restarting the epoch space.
//!
//! [ShuffleExchangeSequence] is an epoch-local event position. Sequence zero
//! means "before the first event", and the first visible event uses sequence one.
//! Consumers should retain an epoch and its sequence together as a
//! [ShuffleExchangeCursor] rather than updating the two values independently.
//!
//! These types intentionally contain no scheduler state, transport contracts,
//! or recovery policy. They are shared vocabulary for those layers.

use std::num::NonZeroU64;

use crate::JobId;

/// Stable identity of the shuffle output produced by one logical stage.
///
/// Stage attempts are deliberately not part of this identity. A retry of the
/// same logical stage still produces the same exchange. Only invalidation of
/// previously accepted materialized output advances that exchange to a new
/// [ShuffleExchangeEpoch].
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShuffleExchangeId {
    job_id: JobId,
    producer_stage_id: usize,
}

impl ShuffleExchangeId {
    /// Creates the exchange identity for producer_stage_id within job_id.
    pub fn new(job_id: JobId, producer_stage_id: usize) -> Self {
        Self {
            job_id,
            producer_stage_id,
        }
    }

    /// Returns the job that owns this exchange.
    pub fn job_id(&self) -> &JobId {
        &self.job_id
    }

    /// Returns the logical producer stage for this exchange.
    pub fn producer_stage_id(&self) -> usize {
        self.producer_stage_id
    }
}

/// Identity of one valid materialized history of a shuffle exchange.
///
/// An epoch is scoped to one [ShuffleExchangeId]; equal numeric epoch values
/// on different exchanges do not identify the same materialized history.
///
/// Epoch zero is intentionally unrepresentable. This keeps zero available as
/// an "unset" value at serialization boundaries and makes stale/uninitialized
/// handles fail closed instead of accidentally naming a real history.
///
/// For one exchange, epoch values must not be reused while handles to an older
/// history can still exist. Scheduler recovery must preserve that monotonicity
/// or explicitly invalidate every pre-recovery handle before restarting it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShuffleExchangeEpoch(NonZeroU64);

impl ShuffleExchangeEpoch {
    /// Epoch assigned to the first materialized history of an exchange.
    pub const INITIAL: Self = Self(NonZeroU64::MIN);

    /// Creates an epoch from its wire/storage representation.
    ///
    /// Returns None for zero because zero never identifies a valid exchange
    /// history.
    pub fn new(value: u64) -> Option<Self> {
        NonZeroU64::new(value).map(Self)
    }

    /// Returns the numeric representation of this epoch.
    pub const fn get(self) -> u64 {
        self.0.get()
    }

    /// Returns the next epoch, or None if the epoch space is exhausted.
    ///
    /// Exhaustion must be handled explicitly by the owning state machine; epoch
    /// rollover must never wrap back to an older identity.
    pub fn checked_next(self) -> Option<Self> {
        self.get().checked_add(1).and_then(Self::new)
    }
}

/// Monotonic event position within one [ShuffleExchangeEpoch].
///
/// Sequence zero means "before the first event". Visible exchange events start
/// at sequence one. A sequence has no standalone meaning: consumers should pair
/// it with the epoch whose event history it indexes via [ShuffleExchangeCursor].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShuffleExchangeSequence(u64);

impl ShuffleExchangeSequence {
    /// Position before any event in an epoch has been observed.
    pub const INITIAL: Self = Self(0);

    /// Creates a sequence from its numeric representation.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the numeric representation of this sequence.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Returns the next sequence, or None if the sequence space is exhausted.
    ///
    /// Exchange event ordering must never wrap because a resumed consumer could
    /// otherwise mistake a new event for an already-observed one.
    pub fn checked_next(self) -> Option<Self> {
        self.0.checked_add(1).map(Self)
    }
}

/// Consumer position within one materialized history of a shuffle exchange.
///
/// Keeping epoch and sequence in one value prevents subscription and reconnect
/// code from independently refreshing the epoch while accidentally retaining a
/// sequence from an older history. New epochs always start from
/// [ShuffleExchangeSequence::INITIAL]; callers advance a cursor without
/// changing its epoch.
///
/// A cursor is still scoped to the [ShuffleExchangeId] that owns its epoch.
/// The exchange identity remains a separate routing/keying dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShuffleExchangeCursor {
    epoch: ShuffleExchangeEpoch,
    sequence: ShuffleExchangeSequence,
}

impl ShuffleExchangeCursor {
    /// Creates the initial consumer position for epoch.
    pub const fn initial(epoch: ShuffleExchangeEpoch) -> Self {
        Self {
            epoch,
            sequence: ShuffleExchangeSequence::INITIAL,
        }
    }

    /// Returns the materialized-history epoch this cursor belongs to.
    pub const fn epoch(self) -> ShuffleExchangeEpoch {
        self.epoch
    }

    /// Returns the last observed event sequence in this epoch.
    pub const fn sequence(self) -> ShuffleExchangeSequence {
        self.sequence
    }

    /// Advances this cursor by one event without changing its epoch.
    ///
    /// Returns None if the sequence space is exhausted.
    pub fn checked_next(self) -> Option<Self> {
        self.sequence.checked_next().map(|sequence| Self {
            epoch: self.epoch,
            sequence,
        })
    }
}

/// Publication lifecycle of one shuffle exchange epoch.
///
/// Lifecycle construction is intentionally explicit rather than implementing
/// [Default]: accidentally defaulting scheduler state to Open would be a
/// fail-open publication decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShuffleExchangeLifecycle {
    /// Producer tasks may still publish output into this epoch.
    Open,
    /// The epoch is complete and can no longer accept new publications.
    Sealed,
}

impl ShuffleExchangeLifecycle {
    /// Returns whether no further output may be published into this epoch.
    pub const fn is_sealed(self) -> bool {
        matches!(self, Self::Sealed)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn exchange_identity_is_job_and_logical_stage() {
        let first = ShuffleExchangeId::new(JobId::from("job-a"), 3);
        let same = ShuffleExchangeId::new(JobId::from("job-a"), 3);
        let other_stage = ShuffleExchangeId::new(JobId::from("job-a"), 4);
        let other_job = ShuffleExchangeId::new(JobId::from("job-b"), 3);

        assert_eq!(first, same);
        assert_ne!(first, other_stage);
        assert_ne!(first, other_job);

        let ids = HashSet::from([first.clone(), other_stage, other_job]);
        assert!(ids.contains(&first));
        assert_eq!(first.job_id(), &JobId::from("job-a"));
        assert_eq!(first.producer_stage_id(), 3);
    }

    #[test]
    fn epoch_reserves_zero_and_never_wraps() {
        assert_eq!(ShuffleExchangeEpoch::new(0), None);
        assert_eq!(ShuffleExchangeEpoch::INITIAL.get(), 1);
        assert_eq!(
            ShuffleExchangeEpoch::INITIAL
                .checked_next()
                .map(ShuffleExchangeEpoch::get),
            Some(2)
        );

        let last = ShuffleExchangeEpoch::new(u64::MAX).unwrap();
        assert_eq!(last.checked_next(), None);
    }

    #[test]
    fn sequence_starts_before_first_event_and_never_wraps() {
        assert_eq!(
            ShuffleExchangeSequence::default(),
            ShuffleExchangeSequence::INITIAL
        );
        assert_eq!(ShuffleExchangeSequence::INITIAL.get(), 0);
        assert_eq!(
            ShuffleExchangeSequence::INITIAL
                .checked_next()
                .map(ShuffleExchangeSequence::get),
            Some(1)
        );
        assert_eq!(
            ShuffleExchangeSequence::new(u64::MAX).checked_next(),
            None
        );
    }

    #[test]
    fn cursor_keeps_epoch_and_sequence_together() {
        let epoch_one = ShuffleExchangeEpoch::INITIAL;
        let cursor = ShuffleExchangeCursor::initial(epoch_one);

        assert_eq!(cursor.epoch(), epoch_one);
        assert_eq!(cursor.sequence(), ShuffleExchangeSequence::INITIAL);

        let next = cursor.checked_next().unwrap();
        assert_eq!(next.epoch(), epoch_one);
        assert_eq!(next.sequence(), ShuffleExchangeSequence::new(1));

        let epoch_two = epoch_one.checked_next().unwrap();
        let fresh = ShuffleExchangeCursor::initial(epoch_two);
        assert_eq!(fresh.epoch(), epoch_two);
        assert_eq!(fresh.sequence(), ShuffleExchangeSequence::INITIAL);
        assert_ne!(next, fresh);
    }

    #[test]
    fn cursor_never_wraps_sequence() {
        let epoch = ShuffleExchangeEpoch::INITIAL;
        let mut cursor = ShuffleExchangeCursor::initial(epoch);
        cursor.sequence = ShuffleExchangeSequence::new(u64::MAX);

        assert_eq!(cursor.checked_next(), None);
        assert_eq!(cursor.epoch(), epoch);
    }

    #[test]
    fn lifecycle_open_and_sealed_are_explicit() {
        assert!(!ShuffleExchangeLifecycle::Open.is_sealed());
        assert!(ShuffleExchangeLifecycle::Sealed.is_sealed());
    }
}
