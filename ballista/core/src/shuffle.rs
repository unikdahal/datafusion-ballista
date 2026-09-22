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
//! is stable for the lifetime of the job; recovery does not manufacture a new
//! logical exchange. Instead, consumers pin a [`ShuffleExchangeEpoch`], which
//! identifies one valid materialized history of that exchange.
//!
//! [`ShuffleExchangeSequence`] is independent from the epoch. It is an
//! epoch-local cursor for ordered exchange events and starts at zero, meaning
//! that no event has been observed yet. The first visible event uses sequence
//! one.
//!
//! These types intentionally contain no scheduler state, transport contracts,
//! or recovery policy. They are shared vocabulary for those layers.

use std::num::NonZeroU64;

use crate::JobId;

/// Stable identity of the shuffle output produced by one logical stage.
///
/// Stage attempts are deliberately not part of this identity. A retry of the
/// same logical stage still produces the same exchange; changes to the valid
/// materialized history are represented by [`ShuffleExchangeEpoch`].
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShuffleExchangeId {
    job_id: JobId,
    producer_stage_id: usize,
}

impl ShuffleExchangeId {
    /// Creates the exchange identity for `producer_stage_id` within `job_id`.
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
/// Epoch zero is intentionally unrepresentable. This keeps zero available as
/// an "unset" value at serialization boundaries and makes stale/uninitialized
/// handles fail closed instead of accidentally naming a real history.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShuffleExchangeEpoch(NonZeroU64);

impl ShuffleExchangeEpoch {
    /// Epoch assigned to the first materialized history of an exchange.
    pub const INITIAL: Self = Self(NonZeroU64::MIN);

    /// Creates an epoch from its wire/storage representation.
    ///
    /// Returns `None` for zero because zero never identifies a valid exchange
    /// history.
    pub fn new(value: u64) -> Option<Self> {
        NonZeroU64::new(value).map(Self)
    }

    /// Returns the numeric representation of this epoch.
    pub fn get(self) -> u64 {
        self.0.get()
    }

    /// Returns the next epoch, or `None` if the sequence space is exhausted.
    ///
    /// Exhaustion must be handled explicitly by the owning state machine; epoch
    /// rollover must never wrap back to an older identity.
    pub fn checked_next(self) -> Option<Self> {
        self.get().checked_add(1).and_then(Self::new)
    }
}

/// Monotonic event cursor within one [`ShuffleExchangeEpoch`].
///
/// Sequence zero is the initial cursor and means "before the first event".
/// Visible exchange events start at sequence one. A sequence is meaningful
/// only together with the epoch whose event history it indexes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShuffleExchangeSequence(u64);

impl ShuffleExchangeSequence {
    /// Cursor before any event in an epoch has been observed.
    pub const INITIAL: Self = Self(0);

    /// Creates a sequence cursor from its numeric representation.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the numeric representation of this sequence.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Returns the next sequence, or `None` if the sequence space is exhausted.
    ///
    /// Exchange event ordering must never wrap because a resumed consumer could
    /// otherwise mistake a new event for an already-observed one.
    pub fn checked_next(self) -> Option<Self> {
        self.0.checked_add(1).map(Self)
    }
}

/// Publication lifecycle of one shuffle exchange epoch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ShuffleExchangeLifecycle {
    /// Producer tasks may still publish output into this epoch.
    #[default]
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
    fn lifecycle_defaults_open_and_sealed_is_explicit() {
        assert_eq!(
            ShuffleExchangeLifecycle::default(),
            ShuffleExchangeLifecycle::Open
        );
        assert!(!ShuffleExchangeLifecycle::Open.is_sealed());
        assert!(ShuffleExchangeLifecycle::Sealed.is_sealed());
    }
}
