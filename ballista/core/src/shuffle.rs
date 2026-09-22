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
//! independently of the exchange that owns it. The pair is represented by
//! [ShuffleExchangeGeneration], which is the fencing identity for one valid
//! materialized history.
//!
//! For the lifetime of a logical exchange identity, the same epoch value must
//! never identify two different materialized histories. Scheduler reconstruction
//! must therefore recover the current epoch/allocation state, advance to a fresh
//! epoch, or abandon the old logical identity. Merely invalidating known handles
//! is insufficient because delayed messages or disconnected readers may still
//! carry an older generation.
//!
//! [ShuffleExchangeSequence] is an epoch-local event position. Sequence zero
//! means "before the first event", and the first visible event uses sequence one.
//! Consumers retain the generation and observed event position together as a
//! [ShuffleExchangeCursor]. Consumers do not allocate sequence numbers; the
//! scheduler owns that monotonic event space.
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
/// For one logical exchange identity, an epoch value must never be reused for a
/// different materialized history. Scheduler recovery must preserve the current
/// allocation state, advance to a fresh epoch, or replace/abandon the logical
/// exchange identity. Resetting the epoch space after invalidating only known
/// handles is unsafe because stale handles and delayed messages can still exist.
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

/// Fencing identity for one materialized history of a logical shuffle exchange.
///
/// The numeric epoch is intentionally never used as a standalone generation
/// identity because equal epoch values on different exchanges are unrelated.
/// Keeping the logical exchange identity and epoch together prevents routing or
/// reconnect code from accidentally validating a handle against the wrong
/// exchange.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShuffleExchangeGeneration {
    exchange_id: ShuffleExchangeId,
    epoch: ShuffleExchangeEpoch,
}

impl ShuffleExchangeGeneration {
    /// Creates a materialized-history generation for one logical exchange.
    pub fn new(exchange_id: ShuffleExchangeId, epoch: ShuffleExchangeEpoch) -> Self {
        Self { exchange_id, epoch }
    }

    /// Creates the first materialized-history generation for an exchange.
    pub fn initial(exchange_id: ShuffleExchangeId) -> Self {
        Self::new(exchange_id, ShuffleExchangeEpoch::INITIAL)
    }

    /// Returns the logical exchange this generation belongs to.
    pub fn exchange_id(&self) -> &ShuffleExchangeId {
        &self.exchange_id
    }

    /// Returns the epoch that fences this materialized history.
    pub const fn epoch(&self) -> ShuffleExchangeEpoch {
        self.epoch
    }
}

/// Consumer position within one materialized history of a shuffle exchange.
///
/// A cursor carries the complete generation identity together with the last
/// scheduler-issued event sequence observed by a consumer. This prevents
/// subscription and reconnect code from pairing a sequence from one history
/// with another exchange or epoch.
///
/// Consumers may reconstruct a cursor at any observed sequence, for example
/// when decoding a reconnect request. They must not generate sequence numbers
/// themselves: sequence allocation belongs to the scheduler's exchange state.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShuffleExchangeCursor {
    generation: ShuffleExchangeGeneration,
    sequence: ShuffleExchangeSequence,
}

impl ShuffleExchangeCursor {
    /// Reconstructs a cursor at an already-observed scheduler event position.
    pub fn new(
        generation: ShuffleExchangeGeneration,
        sequence: ShuffleExchangeSequence,
    ) -> Self {
        Self {
            generation,
            sequence,
        }
    }

    /// Creates the initial consumer position for a materialized generation.
    pub fn initial(generation: ShuffleExchangeGeneration) -> Self {
        Self::new(generation, ShuffleExchangeSequence::INITIAL)
    }

    /// Returns the materialized generation this cursor belongs to.
    pub fn generation(&self) -> &ShuffleExchangeGeneration {
        &self.generation
    }

    /// Returns the logical exchange this cursor belongs to.
    pub fn exchange_id(&self) -> &ShuffleExchangeId {
        self.generation.exchange_id()
    }

    /// Returns the materialized-history epoch this cursor belongs to.
    pub const fn epoch(&self) -> ShuffleExchangeEpoch {
        self.generation.epoch()
    }

    /// Returns the last observed scheduler-issued event sequence.
    pub const fn sequence(&self) -> ShuffleExchangeSequence {
        self.sequence
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
    fn generation_binds_exchange_and_epoch() {
        let exchange_a = ShuffleExchangeId::new(JobId::from("job-a"), 3);
        let exchange_b = ShuffleExchangeId::new(JobId::from("job-a"), 4);
        let epoch_one = ShuffleExchangeEpoch::INITIAL;
        let epoch_two = epoch_one.checked_next().unwrap();

        let a1 = ShuffleExchangeGeneration::new(exchange_a.clone(), epoch_one);
        let a2 = ShuffleExchangeGeneration::new(exchange_a.clone(), epoch_two);
        let b1 = ShuffleExchangeGeneration::new(exchange_b, epoch_one);

        assert_eq!(a1.exchange_id(), &exchange_a);
        assert_eq!(a1.epoch(), epoch_one);
        assert_ne!(a1, a2);
        assert_ne!(a1, b1);
        assert_eq!(
            ShuffleExchangeGeneration::initial(exchange_a),
            ShuffleExchangeGeneration::new(
                ShuffleExchangeId::new(JobId::from("job-a"), 3),
                ShuffleExchangeEpoch::INITIAL,
            )
        );
    }

    #[test]
    fn cursor_reconstructs_arbitrary_observed_position() {
        let exchange = ShuffleExchangeId::new(JobId::from("job-a"), 3);
        let generation =
            ShuffleExchangeGeneration::new(exchange.clone(), ShuffleExchangeEpoch::INITIAL);
        let cursor = ShuffleExchangeCursor::new(
            generation.clone(),
            ShuffleExchangeSequence::new(137),
        );

        assert_eq!(cursor.generation(), &generation);
        assert_eq!(cursor.exchange_id(), &exchange);
        assert_eq!(cursor.epoch(), ShuffleExchangeEpoch::INITIAL);
        assert_eq!(cursor.sequence(), ShuffleExchangeSequence::new(137));
    }

    #[test]
    fn cursor_identity_includes_exchange_epoch_and_sequence() {
        let epoch_one = ShuffleExchangeEpoch::INITIAL;
        let epoch_two = epoch_one.checked_next().unwrap();

        let exchange_a = ShuffleExchangeId::new(JobId::from("job-a"), 3);
        let exchange_b = ShuffleExchangeId::new(JobId::from("job-a"), 4);

        let a1 = ShuffleExchangeCursor::new(
            ShuffleExchangeGeneration::new(exchange_a.clone(), epoch_one),
            ShuffleExchangeSequence::new(7),
        );
        let b1 = ShuffleExchangeCursor::new(
            ShuffleExchangeGeneration::new(exchange_b, epoch_one),
            ShuffleExchangeSequence::new(7),
        );
        let a2 = ShuffleExchangeCursor::initial(ShuffleExchangeGeneration::new(
            exchange_a,
            epoch_two,
        ));

        assert_ne!(a1, b1);
        assert_ne!(a1, a2);
        assert_eq!(a2.epoch(), epoch_two);
        assert_eq!(a2.sequence(), ShuffleExchangeSequence::INITIAL);
    }

    #[test]
    fn lifecycle_open_and_sealed_are_explicit() {
        assert!(!ShuffleExchangeLifecycle::Open.is_sealed());
        assert!(ShuffleExchangeLifecycle::Sealed.is_sealed());
    }
}
