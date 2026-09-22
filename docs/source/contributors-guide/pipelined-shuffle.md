<!--
 Licensed to the Apache Software Foundation (ASF) under one
 or more contributor license agreements.  See the NOTICE file
 distributed with this work for additional information
 regarding copyright ownership.  The ASF licenses this file
 to you under the Apache License, Version 2.0 (the
 "License"); you may not use this file except in compliance
 with the License.  You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing,
 software distributed under the License is distributed on an
 "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 KIND, either express or implied.  See the License for the
 specific language governing permissions and limitations
 under the License.
-->

# Pipelined materialized shuffle

Experimental pipelining overlaps intermediate static stages with the producer
tail. It is disabled by default. Enable `ballista.shuffle.pipelined.enabled` and
disable `ballista.planner.adaptive.enabled` to opt in. AQE, broadcast, range,
coalesced readers, and the externally visible final stage retain their barriers.

The scheduler publishes locations only after accepting a producer task's success.
All outputs from that task share a publication version. A reader pins one
generation, discovers snapshots and deltas through scheduler RPCs, and delegates
physical reads to the existing shuffle fetch engine. Neither shuffle file formats
nor Flight data transport change. Empty producing snapshots mean wait; only a
sealed and drained history permits EOF. Optimizer statistics remain unknown.

Both built-in binders allocate normal work across jobs first. Tail work uses
remaining executor vcores only when each unsealed producer has committed data and
no pending partitions. A producer retry revokes its consumers. Lost committed
output rolls the generation forward while preserving surviving files. All affected
consumers roll back through the existing cancellation machinery. Successful
consumer reports are held until their pinned inputs seal, including operators
that stop reading early (for example LIMIT).

The current scheduler backend is in memory. A scheduler restart closes the live
job and requires job resubmission; generations are not restored from persistence.
Custom distribution policies must explicitly call the admission-aware graph
interface to support early scheduling.

## Review stack

1. `shuffle/01-input-state`: atomic publication, versions, seal, rollover.
2. `shuffle/02-metadata-protocol`: snapshot and bounded long polling.
3. `shuffle/03-pipelined-reader`: discovery, shared fetch path, runtime client, serde.
4. `shuffle/04-task-restriction`: local-to-upstream partition mapping.
5. `shuffle/05-stage-resolution`: feature flag and separate planning path.
6. `shuffle/06-tail-scheduling`: normal-first admission using spare capacity.
7. `shuffle/07-recovery`: revocation, invalidation, fan-out rollback, success barrier.
8. `shuffle/08-validation`: regression coverage and documentation.
9. `shuffle/review-hardening`: empty-publication recovery, task-report ownership,
   exact replay handling, bounded transport retries, and executable job-lifecycle tests.

Validation runs in the `Pipelined shuffle validation` GitHub Actions workflow.
Pushes to the shuffle review branches rerun validation automatically; a manual
`stack` run checks every review branch independently. The workflow checks
formatting, runs core and scheduler unit tests, compiles executor targets, and,
once present on a branch, runs the end-to-end validation.

The end-to-end tests compare feature-off/on results for local and forced Flight
fetches and inspect reader metrics to prove at least one feature-on reader polled
before producer seal. Recovery coverage then removes the executor that owns
committed output while a pipelined reader is active, requires the pinned
generation to become invalid, and verifies the retried query returns every row
exactly once.

The review follow-up also covers loss of empty publications, duplicate terminal
reports, wrong-executor reports and refunds, exact versus conflicting metadata
replays, retry cursors, and cancellation of an in-flight metadata request. After
the reader's short transport retry loop is exhausted, temporary unavailability
is reported as retryable task I/O and remains bounded by the scheduler retry policy.

The ignored latency experiment uses non-blocking deterministic delay injection and
reports repeated samples plus medians. It is diagnostic smoke coverage, not
benchmark evidence. Before production rollout, measure query latency and idle
vcore time under controlled, repeatable load with identical input and capacity.
The flag remains experimental until broader failure testing and controlled
performance measurements establish the benefit.
