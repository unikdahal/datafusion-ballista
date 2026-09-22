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

use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::fmt::{Debug, Formatter};
use std::iter::FromIterator;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use datafusion::physical_plan::display::DisplayableExecutionPlan;
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanVisitor, accept};
use datafusion::prelude::SessionConfig;
use log::{debug, error, info, warn};

use ballista_core::JobId;
use ballista_core::error::{BallistaError, Result};
use ballista_core::execution_plans::{
    RangeShuffleWriterExec, ShuffleWriter, ShuffleWriterExec, SortShuffleWriterExec,
    UnresolvedShuffleExec,
};
use ballista_core::extension::SessionConfigExt;
use ballista_core::serde::protobuf::failed_task::FailedReason;
use ballista_core::serde::protobuf::job_status::Status;
use ballista_core::serde::protobuf::{FailedJob, ShuffleWritePartition, job_status};
use ballista_core::serde::protobuf::{
    FailedTask, JobStatus, ResultLost, RunningJob, SuccessfulJob, SuccessfulTask,
    TaskStatus,
};
use ballista_core::serde::protobuf::{RunningTask, task_status};
use ballista_core::serde::scheduler::{
    ExecutorMetadata, PartitionId, PartitionLocation, PartitionStats, TaskKey,
};

use crate::display::print_stage_metrics;
use crate::planner::DistributedPlanner;
use crate::scheduler_server::event::QueryStageSchedulerEvent;
use crate::scheduler_server::timestamp_millis;
use ballista_core::execution_plans::log_merged_runtime_stats;

use crate::state::execution_stage::RunningStage;
pub(crate) use crate::state::execution_stage::{
    ExecutionStage, ResolvedStage, StageOutput, TaskInfo, UnresolvedStage,
};
use crate::state::task_manager::UpdatedStages;

/// Boxed [ExecutionGraph]
pub type ExecutionGraphBox = Box<dyn ExecutionGraph + Send + Sync>;

// Recovery semantics are present in this layer, so the experimental feature
// may be enabled in production builds when the session flag opts in.
const PIPELINED_SHUFFLE_RECOVERY_READY: bool = true;

/// Represents the DAG for a distributed query plan.
///
/// A distributed query plan consists of a set of stages which must be executed sequentially.
///
/// Each stage consists of a set of partitions which can be executed in parallel, where each partition
/// represents a `Task`, which is the basic unit of scheduling in Ballista.
///
/// As an example, consider a SQL query which performs a simple aggregation:
///
/// `SELECT id, SUM(gmv) FROM some_table GROUP BY id`
///
/// This will produce a DataFusion execution plan that looks something like
///
/// ```text
///   CoalesceBatchesExec: target_batch_size=4096
///     RepartitionExec: partitioning=Hash([Column { name: "id", index: 0 }], 4)
///       AggregateExec: mode=Partial, gby=[id\@0 as id], aggr=[SUM(some_table.gmv)]
///         TableScan: some_table
/// ```
///
/// The Ballista `DistributedPlanner` will turn this into a distributed plan by creating a shuffle
/// boundary (called a "Stage") whenever the underlying plan needs to perform a repartition.
/// In this case we end up with a distributed plan with two stages:
///
/// ```text
/// ExecutionGraph[job_id=job, session_id=session, available_tasks=1, complete=false]
/// =========UnResolvedStage[id=2, children=1]=========
/// Inputs{1: StageOutput { partition_locations: {}, complete: false }}
/// ShuffleWriterExec: None
///   AggregateExec: mode=FinalPartitioned, gby=[id\@0 as id], aggr=[SUM(?table?.gmv)]
///     CoalesceBatchesExec: target_batch_size=4096
///       UnresolvedShuffleExec
/// =========ResolvedStage[id=1, partitions=1]=========
/// ShuffleWriterExec: Some(Hash([Column { name: "id", index: 0 }], 4))
///   AggregateExec: mode=Partial, gby=[id\@0 as id], aggr=[SUM(?table?.gmv)]
///     TableScan: some_table
/// ```
///
/// The DAG structure of this `ExecutionGraph` is encoded in the stages. Each stage's `input` field
/// will indicate which stages it depends on, and each stage's `output_links` will indicate which
/// stage it needs to publish its output to.
///
/// If a stage has `output_links` is empty then it is the final stage in this query, and it should
/// publish its outputs to the `ExecutionGraph`s `output_locations` representing the final query results.
pub trait ExecutionGraph: Debug {
    /// Runtime metadata is absent for graph implementations without pipelining.
    fn shuffle_inputs(
        &self,
    ) -> Option<Arc<parking_lot::Mutex<super::shuffle_input::ShuffleInputRegistry>>> {
        None
    }

    /// Returns the job ID for this execution graph.
    fn job_id(&self) -> &JobId;

    /// Returns the job name for this execution graph.
    fn job_name(&self) -> &str;

    /// Returns the session ID associated with this job.
    fn session_id(&self) -> &str;

    /// Returns the session config associated with this job.
    fn session_config(&self) -> Arc<SessionConfig>;

    /// Returns the current status of the job.
    fn status(&self) -> &JobStatus;

    /// Returns the logical plan as a string, if captured at submission time.
    fn logical_plan(&self) -> Option<&str>;

    /// Returns the physical plan as a string, if captured at submission time.
    fn physical_plan(&self) -> Arc<dyn ExecutionPlan>;

    /// Returns the timestamp when this job started execution.
    fn start_time(&self) -> u64;

    /// Returns the timestamp when this job started execution.
    fn end_time(&self) -> u64;

    /// Number of completed stages
    fn completed_stages(&self) -> usize;

    /// An ExecutionGraph is successful if all its stages are successful
    fn is_successful(&self) -> bool;

    /// Revive the execution graph by converting the resolved stages to running stages
    /// If any stages are converted, return true; else false.
    fn revive(&mut self) -> bool;

    /// Update task statuses and task metrics in the graph.
    /// This will also push shuffle partitions to their respective shuffle read stages.
    fn update_task_status(
        &mut self,
        executor: &ExecutorMetadata,
        task_statuses: Vec<TaskStatus>,
        max_task_failures: usize,
        max_stage_failures: usize,
    ) -> Result<Vec<QueryStageSchedulerEvent>>;

    /// Returns all the currently running stage IDs.
    fn running_stages(&self) -> Vec<usize>;

    /// Returns all currently running tasks along with the executor ID on which they are assigned.
    fn running_tasks(&self) -> Vec<RunningTaskInfo>;

    /// Returns the total number of tasks in this plan that are ready for scheduling.
    fn available_tasks(&self) -> usize;

    /// Fetches a running stage that has available tasks, excluding stages in the blacklist.
    ///
    /// Returns a mutable reference to the running stage if a suitable
    /// stage is found. task_id is assigned per-stage as
    /// `task_infos.len()` at bind time (see `bind_one`), so no external
    /// generator is needed.
    fn fetch_running_stage(&mut self, black_list: &[usize]) -> Option<&mut RunningStage>;

    /// Normal work is bound across all jobs before revocable producer-tail work.
    fn fetch_running_stage_for_admission(
        &mut self,
        black_list: &[usize],
        tail: bool,
    ) -> Option<&mut RunningStage> {
        if tail {
            None
        } else {
            self.fetch_running_stage(black_list)
        }
    }

    /// Updates the job status.
    fn update_status(&mut self, status: JobStatus);

    /// Returns the output partition locations for the final stage results.
    fn output_locations(&self) -> Vec<PartitionLocation>;

    /// Reset running and successful stages on a given executor
    /// This will first check the unresolved/resolved/running stages and reset the running tasks and successful tasks.
    /// Then it will check the successful stage and whether there are running parent stages need to read shuffle from it.
    /// If yes, reset the successful tasks and roll back the resolved shuffle recursively.
    ///
    /// Returns the reset stage ids and running tasks should be killed
    fn reset_stages_on_lost_executor(
        &mut self,
        executor_id: &str,
    ) -> Result<(HashSet<usize>, Vec<RunningTaskInfo>)>;

    /// Converts an unresolved stage to resolved state.
    ///
    /// Returns true if the stage was successfully resolved, false if the stage
    /// was not found or not in unresolved state.
    fn resolve_stage(&mut self, stage_id: usize) -> Result<bool>;

    /// Converts a running stage to successful state.
    ///
    /// Returns true if the stage was successfully marked as complete.
    fn succeed_stage(&mut self, stage_id: usize) -> bool;

    /// Converts a running stage to failed state with the given error message.
    ///
    /// Returns true if the stage was found and marked as failed.
    fn fail_stage(&mut self, stage_id: usize, err_msg: String) -> bool;

    /// Convert running stage to be unresolved,
    /// Returns a Vec of RunningTaskInfo for running tasks in this stage.
    fn rollback_running_stage(
        &mut self,
        stage_id: usize,
        failure_reasons: HashSet<String>,
    ) -> Result<Vec<RunningTaskInfo>>;

    /// Convert resolved stage to be unresolved
    fn rollback_resolved_stage(&mut self, stage_id: usize) -> Result<bool>;

    /// Converts a successful stage back to running state for re-execution.
    ///
    /// This is used when some outputs from the stage have been lost and tasks
    /// need to be re-run.
    fn rerun_successful_stage(&mut self, stage_id: usize) -> bool;

    /// fail job with error message
    fn fail_job(&mut self, error: String);

    /// Abort a running job: fail it, transition every running stage to Failed,
    /// and return the in-flight tasks that should be cancelled. Used for both the
    /// failure and cancellation teardown paths.
    fn abort_running(&mut self, error: String) -> Vec<RunningTaskInfo> {
        let running_tasks = self.running_tasks();
        self.fail_job(error.clone());
        for stage_id in self.running_stages() {
            self.fail_stage(stage_id, error.clone());
        }
        running_tasks
    }

    /// Marks the job as successfully completed.
    ///
    /// This should only be called after all stages have completed successfully.
    /// Returns an error if the job is not in a successful state.
    fn succeed_job(&mut self) -> Result<()>;

    /// Exposes executions stages and stage id's
    fn stages(&self) -> &HashMap<usize, ExecutionStage>;

    /// Stage ids of all non-final (intermediate) stages — those whose
    /// `output_links` is non-empty. The final stage(s) are excluded.
    fn intermediate_stage_ids(&self) -> Vec<u32> {
        self.stages()
            .iter()
            .filter(|(_, stage)| !stage.output_links().is_empty())
            .map(|(stage_id, _)| *stage_id as u32)
            .collect()
    }

    /// Vcores a task consumed from the executor's budget at bind time.
    /// Usually equals `global_input_partition_ids.len()`, but for collapse
    /// tasks that monopolize the executor it is capped at the budget
    /// available when they were bound. Returns `None` if the stage or task
    /// is unknown — e.g. the stage was evicted or the `task_id` (append
    /// slot in `task_infos`) is out of range.
    fn task_vcores(&self, stage_id: usize, task_id: usize) -> Option<u32> {
        self.stages()
            .get(&stage_id)
            .and_then(|s| s.task_infos())
            .and_then(|infos| infos.get(task_id))
            .map(|ti| ti.vcores_consumed)
    }

    /// Refund completed work, including tasks retired by stage rollback.
    fn release_task_vcores(&mut self, _executor_id: &str, status: &TaskStatus) -> u32 {
        self.task_vcores(status.stage_id as usize, status.task_id as usize)
            .unwrap_or(0)
    }

    /// returns next task to run
    /// (used for testing only)
    #[cfg(test)]
    fn pop_next_task(&mut self, executor_id: &str) -> Result<Option<TaskDescription>>;

    /// Returns the total number of stages in this execution graph.
    fn stage_count(&self) -> usize;

    /// Clones execution graph
    fn cloned(&self) -> ExecutionGraphBox;
}

/// [ExecutionGraph] implementation which generates
/// all stages on job submission time
#[derive(Clone)]
pub struct StaticExecutionGraph {
    retired_tasks: HashMap<usize, Vec<TaskInfo>>,
    retired_vcores: HashMap<(usize, usize, usize), (String, u32)>,
    refunded_tasks: HashSet<(usize, usize, usize)>,
    /// Operators such as LIMIT may finish without draining their inputs. Hold
    /// their success until all pinned histories seal, so it cannot escape recovery.
    deferred_successes: Vec<(ExecutorMetadata, TaskStatus)>,
    /// Scheduler-local recovery history. The current JobState implementation
    /// does not reacquire running execution graphs after restart; any future
    /// persistent/HA JobState must persist/rebuild this registry before it can
    /// safely resume a pipelined job.
    shuffle_inputs: Arc<parking_lot::Mutex<super::shuffle_input::ShuffleInputRegistry>>,
    /// A deterministic pipelined-plan rewrite failure disables further tail
    /// plan construction for this job. Existing valid tail stages retain their
    /// explicit admission semantics while unresolved consumers fall back to
    /// the blocking barrier.
    pipelined_resolution_error: Option<String>,
    /// Curator scheduler name. Can be `None` is `ExecutionGraph` is not currently curated by any scheduler
    #[allow(dead_code)] // not used at the moment, will be used later
    scheduler_id: Option<String>,
    /// ID for this job
    job_id: JobId,
    /// Job name, can be empty string
    job_name: String,
    /// Session ID for this job
    session_id: String,
    /// Status of this job
    status: JobStatus,
    /// Timestamp of when this job was submitted
    queued_at: u64,
    /// Job start time
    start_time: u64,
    /// Job end time
    end_time: u64,
    /// Map from Stage ID -> ExecutionStage
    stages: HashMap<usize, ExecutionStage>,
    /// Locations of this `ExecutionGraph` final output locations
    output_locations: Vec<PartitionLocation>,
    /// Failed stage attempts, record the failed stage attempts to limit the retry times.
    /// Map from Stage ID -> Set<Stage_ATTPMPT_NUM>
    failed_stage_attempts: HashMap<usize, HashSet<usize>>,
    /// Session config for this job
    session_config: Arc<SessionConfig>,
    /// Logical plan as a human-readable string, captured at submission time.
    logical_plan: Option<String>,
    /// Physical plan as a human-readable string, captured at submission time.
    physical_plan: Arc<dyn ExecutionPlan>,
}

/// Information about a currently running task.
///
/// Used to track tasks that are in progress and may need to be cancelled
/// when an executor is lost or a job is cancelled.
#[derive(Clone, Debug)]
pub struct RunningTaskInfo {
    /// Append-order slot of this task in `RunningStage.task_infos`;
    /// `(job_id, stage_id, task_id)` is globally unique.
    pub task_id: usize,
    /// The job ID this task belongs to.
    pub job_id: JobId,
    /// The stage ID this task belongs to.
    pub stage_id: usize,
    /// The executor ID where this task is running.
    pub executor_id: String,
}

/// Single source of truth for shuffle shapes that are safe for static
/// pipelining. Keeping the matrix explicit prevents later planner additions
/// from accidentally widening admission.
fn pipelined_shape_eligible(
    broadcast: bool,
    coalesced: bool,
    range_reader: bool,
    range_writer: bool,
) -> bool {
    !broadcast && !coalesced && !range_reader && !range_writer
}

fn pipelined_plan_eligible(plan: &dyn ExecutionPlan) -> bool {
    if let Some(reader) = plan.downcast_ref::<UnresolvedShuffleExec>() {
        return pipelined_shape_eligible(
            reader.broadcast,
            reader.coalesce.is_some(),
            matches!(
                reader.properties().partitioning,
                datafusion::physical_plan::Partitioning::Range(_)
            ),
            false,
        );
    }
    if plan.is::<RangeShuffleWriterExec>() {
        return pipelined_shape_eligible(false, false, false, true);
    }
    plan.children()
        .iter()
        .all(|child| pipelined_plan_eligible(child.as_ref()))
}

impl StaticExecutionGraph {
    /// Restore retired append-only task slots when a revoked tail attempt is
    /// rebuilt. The new attempt intentionally keeps *all* input partitions
    /// pending: the retired TaskInfo entries reserve their old task IDs only;
    /// they are not evidence that any partition completed in the new attempt.
    fn restore_task_identity(&self, stage: &mut RunningStage) {
        if let Some(retired) = self.retired_tasks.get(&stage.stage_id) {
            stage.task_infos = retired.clone();
        }
    }
    fn reconcile_pipelined(&mut self) -> Result<Vec<RunningTaskInfo>> {
        use super::execution_stage::StageAdmission;
        if !self.pipelining_enabled() {
            return Ok(vec![]);
        }
        let mut cancel = vec![];
        loop {
            let mut invalidated = HashSet::new();
            for (id, stage) in &self.stages {
                let accepted = stage
                    .task_infos()
                    .unwrap_or_default()
                    .iter()
                    .filter_map(|info| {
                        matches!(info.task_status, task_status::Status::Successful(_))
                            .then_some(info.task_id)
                    })
                    .collect();
                if self.shuffle_inputs.lock().retain_tasks(
                    &self.job_id,
                    *id,
                    &accepted,
                )? {
                    invalidated.insert(*id);
                }
            }
            let rollback: Vec<_> = self
                .stages
                .iter()
                .filter_map(|(id, stage)| {
                    let (inputs, admission) = match stage {
                        ExecutionStage::Running(s) => (&s.inputs, s.admission.clone()),
                        ExecutionStage::Resolved(s) => (&s.inputs, s.admission.clone()),
                        ExecutionStage::Successful(s) => {
                            (&s.inputs, StageAdmission::from_plan(&s.plan))
                        }
                        _ => return None,
                    };
                    let history_lost =
                        inputs.keys().any(|producer| invalidated.contains(producer));
                    let revoked = match admission {
                        StageAdmission::Normal => false,
                        StageAdmission::TailPipelined(handles) => {
                            !self.input_histories_ready(&handles, false)
                        }
                    };
                    (history_lost || revoked).then_some(*id)
                })
                .collect();
            for id in &rollback {
                if matches!(self.stages.get(id), Some(ExecutionStage::Successful(_))) {
                    self.rerun_successful_stage(*id);
                }
                match self.stages.get(id) {
                    Some(ExecutionStage::Running(_)) => {
                        cancel.extend(self.rollback_running_stage(*id, HashSet::new())?);
                    }
                    Some(ExecutionStage::Resolved(_)) => {
                        self.rollback_resolved_stage(*id)?;
                    }
                    _ => {}
                }
                if self
                    .stages
                    .get(id)
                    .is_some_and(|stage| stage.output_links().is_empty())
                {
                    self.output_locations.clear();
                }
                self.deferred_successes
                    .retain(|(_, status)| status.stage_id as usize != *id);
            }
            // Refresh dependency mirrors from the authoritative committed history.
            let registry = self.shuffle_inputs.lock();
            for stage in self.stages.values_mut() {
                let inputs = match stage {
                    ExecutionStage::UnResolved(s) => &mut s.inputs,
                    ExecutionStage::Resolved(s) => &mut s.inputs,
                    ExecutionStage::Running(s) => &mut s.inputs,
                    _ => continue,
                };
                for (producer, output) in inputs {
                    if let Some(current) = registry.stage_output(&self.job_id, *producer)
                    {
                        *output = current;
                    }
                }
            }
            drop(registry);
            if rollback.is_empty() {
                break;
            }
        }
        // A revoked consumer whose inputs have since sealed uses the blocking path.
        let ready: Vec<_> = self
            .stages
            .iter()
            .filter_map(|(id, stage)| {
                matches!(stage, ExecutionStage::UnResolved(s) if s.resolvable())
                    .then_some(*id)
            })
            .collect();
        for id in ready {
            self.resolve_stage(id)?;
        }
        Ok(cancel)
    }
    /// Check whether every pinned producer history is safe for this admission.
    ///
    /// Atomicity relies on the caller holding the execution graph's exclusive
    /// write guard while this method observes both registry history and each
    /// producer's pending queue. The registry mutex alone is not sufficient:
    /// task-status processing mutates the history and stage state together.
    /// Keep this helper private unless that locking contract is preserved.
    fn input_histories_ready(
        &self,
        handles: &[ballista_core::serde::protobuf::ShuffleInputHandle],
        require_sealed: bool,
    ) -> bool {
        use super::shuffle_input::ShuffleInputLifecycle;
        let registry = self.shuffle_inputs.lock();
        handles.iter().all(|handle| {
            let Some(state) = registry.get(&self.job_id, handle.stage_id as usize) else {
                return false;
            };
            if state.generation() != handle.generation {
                return false;
            }
            if state.lifecycle() == ShuffleInputLifecycle::Sealed {
                return true;
            }
            if require_sealed || !state.has_committed_input() {
                return false;
            }
            matches!(
                self.stages.get(&(handle.stage_id as usize)),
                Some(ExecutionStage::Running(producer)) if producer.pending.is_empty()
            )
        })
    }

    /// Resolve all currently eligible tail consumers. No candidate is a normal
    /// temporary state and returns Ok(()); an Err means plan construction
    /// itself failed and is therefore deterministic for this graph state.
    fn resolve_tail_stages(&mut self) -> Result<()> {
        let candidates: Vec<_> = self
            .stages
            .iter()
            .filter_map(|(id, stage)| {
                let ExecutionStage::UnResolved(stage) = stage else {
                    return None;
                };
                if stage.resolvable() {
                    return None;
                }
                let handles = self.pipelined_handles(stage)?;
                let input_refs: Vec<_> = handles.values().cloned().collect();
                self.input_histories_ready(&input_refs, false)
                    .then_some((*id, handles))
            })
            .collect();

        // Build every replacement before mutating the graph. A deterministic
        // rewrite failure must not leave a prefix of candidates converted to
        // tail stages while the caller believes tail construction failed.
        let mut resolved = Vec::with_capacity(candidates.len());
        for (id, handles) in candidates {
            let Some(ExecutionStage::UnResolved(stage)) = self.stages.get(&id) else {
                continue;
            };
            let mut running = stage.to_pipelined_resolved(&handles)?.to_running();
            self.restore_task_identity(&mut running);
            resolved.push((id, running));
        }
        for (id, stage) in resolved {
            self.stages.insert(id, ExecutionStage::Running(stage));
        }
        Ok(())
    }

    fn running_stage_for_admission_id(
        &self,
        black_list: &[usize],
        tail: bool,
    ) -> Option<usize> {
        use super::execution_stage::StageAdmission;

        self.stages
            .iter()
            .filter_map(|(id, stage)| {
                let ExecutionStage::Running(stage) = stage else {
                    return None;
                };
                if black_list.contains(id) || stage.pending.is_empty() {
                    return None;
                }
                let eligible = match &stage.admission {
                    StageAdmission::Normal => !tail,
                    StageAdmission::TailPipelined(inputs) => {
                        if self.input_histories_ready(inputs, true) {
                            !tail
                        } else {
                            tail && self.input_histories_ready(inputs, false)
                        }
                    }
                };
                eligible.then_some(*id)
            })
            .min()
    }

    fn pipelining_enabled(&self) -> bool {
        PIPELINED_SHUFFLE_RECOVERY_READY
            && self.session_config.ballista_shuffle_pipelined_enabled()
            && !self
                .session_config
                .ballista_adaptive_query_planner_enabled()
    }

    /// Eligibility is structural. Admission separately checks producer pending work.
    fn pipelined_handles(
        &self,
        stage: &UnresolvedStage,
    ) -> Option<HashMap<usize, ballista_core::serde::protobuf::ShuffleInputHandle>> {
        if !self.pipelining_enabled()
            || stage.output_links.is_empty()
            || stage.inputs.is_empty()
        {
            return None;
        }
        if !pipelined_plan_eligible(stage.plan.as_ref()) {
            return None;
        }
        let registry = self.shuffle_inputs.lock();
        stage
            .inputs
            .keys()
            .map(|producer| {
                let producer_plan = self.stages.get(producer)?.plan();
                if producer_plan.is::<RangeShuffleWriterExec>() {
                    return None;
                }
                let state = registry.get(&self.job_id, *producer)?;
                Some((
                    *producer,
                    ballista_core::serde::protobuf::ShuffleInputHandle {
                        job_id: self.job_id.to_string(),
                        stage_id: *producer as u32,
                        generation: state.generation(),
                    },
                ))
            })
            .collect()
    }
    /// Creates a new `ExecutionGraph` from a physical execution plan.
    ///
    /// This will use the `DistributedPlanner` to break the plan into stages
    /// and build the DAG structure needed for distributed execution.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        scheduler_id: &str,
        job_id: &JobId,
        job_name: &str,
        session_id: &str,
        plan: Arc<dyn ExecutionPlan>,
        queued_at: u64,
        session_config: Arc<SessionConfig>,
        planner: &mut dyn DistributedPlanner,
        logical_plan: Option<String>,
    ) -> Result<Self> {
        let shuffle_stages =
            planner.plan_query_stages(job_id, plan.clone(), session_config.options())?;

        let builder = ExecutionStageBuilder::new(session_config.clone());
        let stages = builder.build(shuffle_stages)?;

        let pipelining_requested = session_config.ballista_shuffle_pipelined_enabled();
        let adaptive_enabled = session_config.ballista_adaptive_query_planner_enabled();
        if pipelining_requested && !PIPELINED_SHUFFLE_RECOVERY_READY {
            warn!(
                "ballista.shuffle.pipelined.enabled=true is unavailable in this build; \
                 the recovery layer must be included before tail scheduling can be enabled"
            );
        } else if pipelining_requested && adaptive_enabled {
            warn!(
                "ballista.shuffle.pipelined.enabled=true is ignored because \
                 ballista.planner.adaptive.enabled=true; disable adaptive planning \
                 to enable experimental static pipelined shuffle"
            );
        }

        let mut shuffle_inputs = super::shuffle_input::ShuffleInputRegistry::default();
        if PIPELINED_SHUFFLE_RECOVERY_READY && pipelining_requested && !adaptive_enabled {
            for stage_id in stages.keys() {
                shuffle_inputs.create(job_id, *stage_id);
            }
        }

        let started_at = timestamp_millis();

        Ok(Self {
            retired_tasks: HashMap::new(),
            retired_vcores: HashMap::new(),
            refunded_tasks: HashSet::new(),
            deferred_successes: vec![],
            shuffle_inputs: Arc::new(parking_lot::Mutex::new(shuffle_inputs)),
            pipelined_resolution_error: None,
            scheduler_id: Some(scheduler_id.to_string()),
            job_id: job_id.to_owned(),
            job_name: job_name.to_owned(),
            session_id: session_id.to_string(),

            status: JobStatus {
                job_id: job_id.to_string(),
                job_name: job_name.to_string(),
                status: Some(Status::Running(RunningJob {
                    queued_at,
                    started_at,
                    scheduler: scheduler_id.to_string(),
                })),
            },
            queued_at,
            start_time: started_at,
            end_time: 0,
            stages,
            output_locations: vec![],
            failed_stage_attempts: HashMap::new(),
            session_config,
            logical_plan,
            physical_plan: plan,
        })
    }

    /// Processing stage status update after task status changing
    fn processing_stages_update(
        &mut self,
        updated_stages: UpdatedStages,
    ) -> Result<Vec<QueryStageSchedulerEvent>> {
        let job_id = self.job_id().to_owned();
        let mut has_resolved = false;
        let mut job_err_msg = "".to_owned();

        for stage_id in updated_stages.resolved_stages {
            self.resolve_stage(stage_id)?;
            has_resolved = true;
        }

        for stage_id in updated_stages.successful_stages {
            self.succeed_stage(stage_id);
        }

        // Fail the stage and also abort the job
        for (stage_id, err_msg) in &updated_stages.failed_stages {
            job_err_msg =
                format!("Job failed due to stage {stage_id} failed: {err_msg}\n");
        }

        let mut events = vec![];
        // Only handle the rollback logic when there are no failed stages
        if updated_stages.failed_stages.is_empty() {
            let mut running_tasks_to_cancel = vec![];
            for (stage_id, failure_reasons) in updated_stages.rollback_running_stages {
                let tasks = self.rollback_running_stage(stage_id, failure_reasons)?;
                running_tasks_to_cancel.extend(tasks);
            }

            for stage_id in updated_stages.resubmit_successful_stages {
                self.rerun_successful_stage(stage_id);
            }

            running_tasks_to_cancel.extend(self.reconcile_pipelined()?);

            if !running_tasks_to_cancel.is_empty() {
                events.push(QueryStageSchedulerEvent::CancelTasks(
                    running_tasks_to_cancel,
                ));
            }
        }

        if !updated_stages.failed_stages.is_empty() {
            info!("Job {job_id} is failed");
            self.fail_job(job_err_msg.clone());
            events.push(QueryStageSchedulerEvent::JobRunningFailed {
                job_id,
                fail_message: job_err_msg,
                queued_at: self.queued_at,
                failed_at: timestamp_millis(),
            });
        } else if self.is_successful() {
            // If this ExecutionGraph is successful, finish it
            debug!("Job {job_id} is success, finalizing job output ...");
            self.succeed_job()?;
            events.push(QueryStageSchedulerEvent::JobFinished {
                job_id,
                queued_at: self.queued_at,
                completed_at: timestamp_millis(),
            });
        } else if has_resolved {
            events.push(QueryStageSchedulerEvent::JobUpdated(job_id))
        }
        Ok(events)
    }

    /// Return a Vec of resolvable stage ids
    fn update_stage_output_links(
        &mut self,
        stage_id: usize,
        is_completed: bool,
        locations: Vec<PartitionLocation>,
        output_links: Vec<usize>,
    ) -> Result<Vec<usize>> {
        let mut resolved_stages = vec![];
        let job_id = &self.job_id;
        if output_links.is_empty() {
            // If `output_links` is empty, then this is a final stage
            self.output_locations.extend(locations);
        } else {
            for link in output_links.iter() {
                // If this is an intermediate stage, we need to push its `PartitionLocation`s to the parent stage
                if let Some(linked_stage) = self.stages.get_mut(link) {
                    if let ExecutionStage::UnResolved(linked_unresolved_stage) =
                        linked_stage
                    {
                        linked_unresolved_stage
                            .add_input_partitions(stage_id, locations.clone())?;

                        // If all tasks for this stage are complete, mark the input complete in the parent stage
                        if is_completed {
                            linked_unresolved_stage.complete_input(stage_id);
                        }

                        // If all input partitions are ready, we can resolve any UnresolvedShuffleExec in the parent stage plan
                        if linked_unresolved_stage.resolvable() {
                            resolved_stages.push(linked_unresolved_stage.stage_id);
                        }
                    } else if let ExecutionStage::Running(stage) = linked_stage
                        && matches!(
                            stage.admission,
                            super::execution_stage::StageAdmission::TailPipelined(_)
                        )
                    {
                        if let Some(input) = stage.inputs.get_mut(&stage_id) {
                            for location in locations.iter().cloned() {
                                input
                                    .partition_locations
                                    .entry(location.partition_id.partition_id)
                                    .or_default()
                                    .push(location);
                            }
                            input.complete = is_completed;
                        }
                    } else if let ExecutionStage::Resolved(stage) = linked_stage
                        && matches!(
                            stage.admission,
                            super::execution_stage::StageAdmission::TailPipelined(_)
                        )
                    {
                        if let Some(input) = stage.inputs.get_mut(&stage_id) {
                            for location in locations.iter().cloned() {
                                input
                                    .partition_locations
                                    .entry(location.partition_id.partition_id)
                                    .or_default()
                                    .push(location);
                            }
                            input.complete = is_completed;
                        }
                    } else {
                        return Err(BallistaError::Internal(format!(
                            "Error updating job {job_id}: The stage {link} as the output link of stage {stage_id}  should be unresolved"
                        )));
                    }
                } else {
                    return Err(BallistaError::Internal(format!(
                        "Error updating job {job_id}: Invalid output link {stage_id} for stage {link}"
                    )));
                }
            }
        }
        Ok(resolved_stages)
    }

    fn get_running_stage_id(&mut self, black_list: &[usize]) -> Option<usize> {
        let mut running_stage_id = self.stages.iter().find_map(|(stage_id, stage)| {
            if black_list.contains(stage_id) {
                None
            } else if let ExecutionStage::Running(stage) = stage {
                if stage.available_tasks() > 0 {
                    Some(*stage_id)
                } else {
                    None
                }
            } else {
                None
            }
        });

        // If no available tasks found in the running stage,
        // try to find a resolved stage and convert it to the running stage
        if running_stage_id.is_none() {
            if self.revive() {
                running_stage_id = self.get_running_stage_id(black_list);
            } else {
                running_stage_id = None;
            }
        }

        running_stage_id
    }

    fn reset_stages_internal(
        &mut self,
        executor_id: &str,
    ) -> Result<(HashSet<usize>, Vec<RunningTaskInfo>)> {
        let job_id = self.job_id.clone();
        // collect the input stages that need to resubmit
        let mut resubmit_inputs: HashSet<usize> = HashSet::new();

        let mut reset_running_stage = HashSet::new();
        let mut rollback_resolved_stages = HashSet::new();
        let mut rollback_running_stages = HashSet::new();
        let mut resubmit_successful_stages = HashSet::new();

        let mut empty_inputs: HashMap<usize, StageOutput> = HashMap::new();
        // check the unresolved, resolved and running stages
        self.stages
            .iter_mut()
            .for_each(|(stage_id, stage)| {
                let stage_inputs = match stage {
                    ExecutionStage::UnResolved(stage) => {
                        &mut stage.inputs
                    }
                    ExecutionStage::Resolved(stage) => {
                        &mut stage.inputs
                    }
                    ExecutionStage::Running(stage) => {
                        let reset = stage.reset_tasks(executor_id);
                        if reset > 0 {
                            warn!(
                        "Reset {reset} tasks for running job/stage {job_id}/{stage_id} on lost Executor {executor_id}"
                        );
                            reset_running_stage.insert(*stage_id);
                        }
                        &mut stage.inputs
                    }
                    _ => &mut empty_inputs
                };

                // For each stage input, check whether there are input locations match that executor
                // and calculate the resubmit input stages if the input stages are successful.
                let mut rollback_stage = false;
                stage_inputs.iter_mut().for_each(|(input_stage_id, stage_output)| {
                    let mut match_found = false;
                    stage_output.partition_locations.iter_mut().for_each(
                        |(_partition, locs)| {
                            let before_len = locs.len();
                            locs.retain(|loc| loc.executor_meta.id != executor_id);
                            if locs.len() < before_len {
                                match_found = true;
                            }
                        },
                    );
                    if match_found {
                        stage_output.complete = false;
                        rollback_stage = true;
                        resubmit_inputs.insert(*input_stage_id);
                    }
                });

                if rollback_stage {
                    match stage {
                        ExecutionStage::Resolved(_) => {
                            rollback_resolved_stages.insert(*stage_id);
                            warn!(
                            "Roll back resolved job/stage {job_id}/{stage_id} and change ShuffleReaderExec back to UnresolvedShuffleExec");
                        }
                        ExecutionStage::Running(_) => {
                            rollback_running_stages.insert(*stage_id);
                            warn!(
                            "Roll back running job/stage {job_id}/{stage_id} and change ShuffleReaderExec back to UnresolvedShuffleExec");
                        }
                        _ => {}
                    }
                }
            });

        // check and reset the successful stages
        if !resubmit_inputs.is_empty() {
            self.stages
                .iter_mut()
                .filter(|(stage_id, _stage)| resubmit_inputs.contains(stage_id))
                .filter_map(|(_stage_id, stage)| {
                    if let ExecutionStage::Successful(success) = stage {
                        Some(success)
                    } else {
                        None
                    }
                })
                .for_each(|stage| {
                    let reset = stage.reset_tasks(executor_id);
                    if reset > 0 {
                        resubmit_successful_stages.insert(stage.stage_id);
                        warn!(
                            "Reset {} tasks for successful job/stage {}/{} on lost Executor {}",
                            reset, job_id, stage.stage_id, executor_id
                        )
                    }
                });
        }

        for stage_id in rollback_resolved_stages.iter() {
            self.rollback_resolved_stage(*stage_id)?;
        }

        let mut all_running_tasks = vec![];
        for stage_id in rollback_running_stages.iter() {
            let tasks = self.rollback_running_stage(
                *stage_id,
                HashSet::from([executor_id.to_owned()]),
            )?;
            all_running_tasks.extend(tasks);
        }

        for stage_id in resubmit_successful_stages.iter() {
            self.rerun_successful_stage(*stage_id);
        }

        let mut reset_stage = HashSet::new();
        reset_stage.extend(reset_running_stage);
        reset_stage.extend(rollback_resolved_stages);
        reset_stage.extend(rollback_running_stages);
        reset_stage.extend(resubmit_successful_stages);
        Ok((reset_stage, all_running_tasks))
    }

    /// Clear the stage failure count for this stage if the stage is finally success
    fn clear_stage_failure(&mut self, stage_id: usize) {
        self.failed_stage_attempts.remove(&stage_id);
    }
}

impl ExecutionGraph for StaticExecutionGraph {
    fn release_task_vcores(&mut self, executor_id: &str, status: &TaskStatus) -> u32 {
        if !self.pipelining_enabled() {
            return self
                .task_vcores(status.stage_id as usize, status.task_id as usize)
                .unwrap_or(0);
        }
        let key = (
            status.stage_id as usize,
            status.stage_attempt_num as usize,
            status.task_id as usize,
        );
        if !matches!(
            status.status,
            Some(task_status::Status::Successful(_) | task_status::Status::Failed(_))
        ) || self.refunded_tasks.contains(&key)
            || status.job_id != self.job_id.as_str()
            || matches!(&status.status,
                Some(task_status::Status::Successful(success)) if success.executor_id != executor_id)
        {
            return 0;
        }
        let current = self.stages.get(&key.0).and_then(|stage| match stage {
            ExecutionStage::Running(stage)
                if stage.accepts_status_from(executor_id, status) =>
            {
                stage.task_infos.get(key.2).map(|task| task.vcores_consumed)
            }
            _ => None,
        });
        let retired = self
            .retired_vcores
            .get(&key)
            .filter(|(owner, _)| owner == executor_id)
            .map(|(_, vcores)| *vcores);
        let Some(vcores) = current.or(retired) else {
            return 0;
        };
        self.retired_vcores.remove(&key);
        self.refunded_tasks.insert(key);
        vcores
    }
    fn shuffle_inputs(
        &self,
    ) -> Option<Arc<parking_lot::Mutex<super::shuffle_input::ShuffleInputRegistry>>> {
        Some(self.shuffle_inputs.clone())
    }
    fn cloned(&self) -> ExecutionGraphBox {
        Box::new(self.clone())
    }

    fn job_id(&self) -> &JobId {
        &self.job_id
    }

    fn job_name(&self) -> &str {
        self.job_name.as_str()
    }

    fn session_id(&self) -> &str {
        self.session_id.as_str()
    }

    fn session_config(&self) -> Arc<SessionConfig> {
        self.session_config.clone()
    }

    fn status(&self) -> &JobStatus {
        &self.status
    }

    fn logical_plan(&self) -> Option<&str> {
        self.logical_plan.as_deref()
    }

    fn physical_plan(&self) -> Arc<dyn ExecutionPlan> {
        self.physical_plan.clone()
    }

    fn start_time(&self) -> u64 {
        self.start_time
    }

    fn end_time(&self) -> u64 {
        self.end_time
    }

    fn completed_stages(&self) -> usize {
        let mut completed_stages = 0;
        for stage in self.stages.values() {
            if let ExecutionStage::Successful(_) = stage {
                completed_stages += 1;
            }
        }
        completed_stages
    }
    /// An ExecutionGraph is successful if all its stages are successful
    fn is_successful(&self) -> bool {
        self.stages
            .values()
            .all(|s| matches!(s, ExecutionStage::Successful(_)))
    }

    // pub fn is_complete(&self) -> bool {
    //     self.stages
    //         .values()
    //         .all(|s| matches!(s, ExecutionStage::Successful(_)))
    // }

    /// Revive the execution graph by converting the resolved stages to running stages
    /// If any stages are converted, return true; else false.
    fn revive(&mut self) -> bool {
        let running_stages = self
            .stages
            .values()
            .filter_map(|stage| {
                if let ExecutionStage::Resolved(resolved_stage) = stage {
                    let mut running = resolved_stage.to_running();
                    self.restore_task_identity(&mut running);
                    Some(running)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        if running_stages.is_empty() {
            false
        } else {
            for running_stage in running_stages {
                self.stages.insert(
                    running_stage.stage_id,
                    ExecutionStage::Running(running_stage),
                );
            }
            true
        }
    }

    /// Update task statuses and task metrics in the graph.
    /// This will also push shuffle partitions to their respective shuffle read stages.
    fn update_task_status(
        &mut self,
        executor: &ExecutorMetadata,
        task_statuses: Vec<TaskStatus>,
        max_task_failures: usize,
        max_stage_failures: usize,
    ) -> Result<Vec<QueryStageSchedulerEvent>> {
        let job_id = self.job_id().to_owned();
        let pipelining_enabled = self.pipelining_enabled();
        // First of all, classify the statuses by stages
        let mut job_task_statuses: HashMap<usize, Vec<TaskStatus>> = HashMap::new();
        for task_status in task_statuses {
            let stage_id = task_status.stage_id as usize;
            if pipelining_enabled
                && (task_status.job_id != self.job_id.as_str()
                    || !matches!(self.stages.get(&stage_id),
                        Some(ExecutionStage::Running(stage))
                            if stage.accepts_status_from(&executor.id, &task_status)))
            {
                continue;
            }
            if matches!(task_status.status, Some(task_status::Status::Successful(_)))
                && let Some(ExecutionStage::Running(stage)) = self.stages.get(&stage_id)
                && let super::execution_stage::StageAdmission::TailPipelined(handles) =
                    &stage.admission
                && !self.input_histories_ready(handles, true)
            {
                if task_status.stage_attempt_num as usize == stage.stage_attempt_num
                    && !self.deferred_successes.iter().any(|(_, existing)| {
                        existing.stage_id == task_status.stage_id
                            && existing.stage_attempt_num == task_status.stage_attempt_num
                            && existing.task_id == task_status.task_id
                    })
                {
                    self.deferred_successes
                        .push((executor.clone(), task_status));
                }
                continue;
            }
            let stage_task_statuses = job_task_statuses.entry(stage_id).or_default();
            stage_task_statuses.push(task_status);
        }

        // Revive before updating due to some updates not saved
        // It will be refined later
        self.revive();

        let current_running_stages: HashSet<usize> =
            HashSet::from_iter(self.running_stages());

        // Copy the failed stage attempts from self
        let mut failed_stage_attempts: HashMap<usize, HashSet<usize>> = HashMap::new();
        for (stage_id, attempts) in self.failed_stage_attempts.iter() {
            failed_stage_attempts
                .insert(*stage_id, HashSet::from_iter(attempts.iter().copied()));
        }

        let mut resolved_stages = HashSet::new();
        let mut successful_stages = HashSet::new();
        let mut failed_stages = HashMap::new();
        let mut rollback_running_stages = HashMap::new();
        // Producer task ids whose already-published shuffle output was found
        // unreadable by a downstream fetch. The producer may still be Running
        // under pipelined execution, or it may already be Successful under the
        // traditional barrier path; defer the state-specific transition until
        // all reports in this batch have been applied.
        let mut lost_materialized_tasks: HashMap<usize, HashSet<usize>> = HashMap::new();

        for (stage_id, stage_task_statuses) in job_task_statuses {
            if let Some(stage) = self.stages.get_mut(&stage_id) {
                if let ExecutionStage::Running(running_stage) = stage {
                    let mut locations = vec![];
                    for task_status in stage_task_statuses.into_iter() {
                        // Recheck after each report: a preceding terminal report
                        // in the same batch may already have retired this slot.
                        if pipelining_enabled
                            && !running_stage
                                .accepts_status_from(&executor.id, &task_status)
                        {
                            continue;
                        }
                        let task_stage_attempt_num =
                            task_status.stage_attempt_num as usize;
                        let invalid_stage_attempt = if pipelining_enabled {
                            task_stage_attempt_num != running_stage.stage_attempt_num
                        } else {
                            // Preserve the pre-feature scheduler contract:
                            // only older attempts are stale in legacy mode.
                            task_stage_attempt_num < running_stage.stage_attempt_num
                        };
                        if invalid_stage_attempt {
                            warn!(
                                "Ignore TaskStatus update with TID {} from Stage {}.{} while current attempt is {}.{}",
                                task_status.task_id,
                                stage_id,
                                task_stage_attempt_num,
                                stage_id,
                                running_stage.stage_attempt_num
                            );
                            continue;
                        }
                        let task_id = task_status.task_id as usize;
                        let Some(previous) = running_stage.task_infos.get(task_id) else {
                            warn!("Ignore unknown shuffle task {stage_id}/{task_id}");
                            continue;
                        };
                        if matches!(
                            previous.task_status,
                            task_status::Status::Successful(_)
                        ) && matches!(
                            task_status.status,
                            Some(task_status::Status::Successful(_))
                        ) {
                            continue;
                        }
                        let task_identity = format!(
                            "TID {}/{}.{}/{}",
                            job_id, stage_id, task_stage_attempt_num, task_id
                        );
                        let task_status_for_update = task_status.clone();
                        let is_success = matches!(
                            task_status.status.as_ref(),
                            Some(task_status::Status::Successful(_))
                        );
                        if !is_success
                            && !running_stage
                                .update_task_info(task_id, task_status_for_update.clone())
                        {
                            continue;
                        }

                        let TaskStatus {
                            status,
                            metrics: operator_metrics,
                            ..
                        } = task_status;

                        if let Some(task_status::Status::Failed(failed_task)) = status {
                            let failed_reason = failed_task.failed_reason;

                            match failed_reason {
                                Some(FailedReason::FetchPartitionError(
                                    fetch_partiton_error,
                                )) => {
                                    let failed_attempts = failed_stage_attempts
                                        .entry(stage_id)
                                        .or_default();
                                    failed_attempts.insert(task_stage_attempt_num);
                                    if failed_attempts.len() < max_stage_failures {
                                        let map_stage_id =
                                            fetch_partiton_error.map_stage_id as usize;
                                        let map_partition_id = fetch_partiton_error
                                            .map_partition_id
                                            as usize;
                                        let executor_id =
                                            fetch_partiton_error.executor_id;

                                        if !failed_stages.is_empty() {
                                            let error_msg = format!(
                                                "Stages was marked failed, ignore FetchPartitionError from task {task_identity}"
                                            );
                                            warn!("{error_msg}");
                                        } else {
                                            // There are different removal strategies here.
                                            // We can choose just remove the map_partition_id in the FetchPartitionError, when resubmit the input stage, there are less tasks
                                            // need to rerun, but this might miss many more bad input partitions, lead to more stage level retries in following.
                                            // Here we choose remove all the bad input partitions which match the same executor id in this single input stage.
                                            // There are other more aggressive approaches, like considering the executor is lost and check all the running stages in this graph.
                                            // Or count the fetch failure number on executor and mark the executor lost globally.
                                            let removed_map_partitions = running_stage
                                                .remove_input_partitions(
                                                    map_stage_id,
                                                    map_partition_id,
                                                    &executor_id,
                                                )?;

                                            let failure_reasons = rollback_running_stages
                                                .entry(stage_id)
                                                .or_insert_with(HashSet::new);
                                            failure_reasons.insert(executor_id);

                                            let lost_tasks = lost_materialized_tasks
                                                .entry(map_stage_id)
                                                .or_default();
                                            lost_tasks.extend(removed_map_partitions);
                                            warn!(
                                                "Need to resubmit the current running Stage {stage_id} and its map Stage {map_stage_id} due to FetchPartitionError from task {task_identity}"
                                            )
                                        }
                                    } else {
                                        let error_msg = format!(
                                            "Stage {} has failed {} times, \
                                            most recent failure reason: {:?}",
                                            stage_id,
                                            max_stage_failures,
                                            failed_task.error
                                        );
                                        error!("{error_msg}");
                                        failed_stages.insert(stage_id, error_msg);
                                    }
                                }
                                Some(FailedReason::ShuffleInputInvalidated(_))
                                | Some(FailedReason::TailAdmissionRevoked(_)) => {
                                    rollback_running_stages
                                        .entry(stage_id)
                                        .or_insert_with(HashSet::new);
                                }
                                Some(FailedReason::ExecutionError(_)) => {
                                    failed_stages.insert(stage_id, failed_task.error);
                                }
                                Some(_) => {
                                    if failed_task.retryable
                                        && failed_task.count_to_failures
                                    {
                                        if running_stage.task_failure_number(task_id)
                                            < max_task_failures
                                        {
                                            // TODO add new struct to track all the failed task infos
                                            // The failure TaskInfo is ignored and set to None here
                                            running_stage.reset_task_info(task_id);
                                        } else {
                                            // Report the *partitions* that hit the failure
                                            // ceiling — task_id (the append slot) isn't
                                            // user-meaningful under the append-only retries
                                            // model (retries get fresh slots), but
                                            // per-partition failure counters are the
                                            // durable identity.
                                            let over_limit: Vec<usize> = running_stage
                                                .task_infos[task_id]
                                                .global_input_partition_ids
                                                .iter()
                                                .copied()
                                                .filter(|p| {
                                                    running_stage.task_failure_numbers[*p]
                                                        >= max_task_failures
                                                })
                                                .collect();
                                            let subject = if over_limit.len() == 1 {
                                                format!("Task {}", over_limit[0])
                                            } else {
                                                format!("Tasks {over_limit:?}")
                                            };
                                            let error_msg = format!(
                                                "{} in Stage {} failed {} times, fail the stage, most recent failure reason: {:?}",
                                                subject,
                                                stage_id,
                                                max_task_failures,
                                                failed_task.error
                                            );
                                            error!("{error_msg}");
                                            failed_stages.insert(stage_id, error_msg);
                                        }
                                    } else if failed_task.retryable {
                                        // TODO add new struct to track all the failed task infos
                                        // The failure TaskInfo is ignored and set to None here
                                        running_stage.reset_task_info(task_id);
                                    }
                                }
                                None => {
                                    let error_msg = format!(
                                        "Task {task_id} in Stage {stage_id} failed with unknown failure reasons, fail the stage"
                                    );
                                    error!("{error_msg}");
                                    failed_stages.insert(stage_id, error_msg);
                                }
                            }
                        } else if let Some(task_status::Status::Successful(
                            successful_task,
                        )) = status
                        {
                            let SuccessfulTask {
                                partitions,
                                runtime_stats,
                                window_state,
                                ..
                            } = successful_task;
                            let mut committed = partition_to_location(
                                &job_id, task_id, stage_id, executor, partitions,
                            );

                            if pipelining_enabled {
                                // Validate graph-local fallible work before
                                // taking the metadata lock. The current
                                // generation is intentionally selected only
                                // after locking so a still-valid in-flight
                                // task can join a generation that rolled over
                                // while it was executing.
                                if !running_stage.can_publish_task_success(
                                    task_id,
                                    &task_status_for_update,
                                ) {
                                    continue;
                                }
                                let prepared_metrics = running_stage
                                    .prepare_task_metrics(task_id, operator_metrics)?;

                                let mut registry = self.shuffle_inputs.lock();
                                let generation = registry
                                    .get(&job_id, stage_id)
                                    .map(|state| state.generation())
                                    .ok_or_else(|| {
                                        BallistaError::Internal(format!(
                                            "pipelined shuffle input missing for job {job_id}, stage {stage_id}"
                                        ))
                                    })?;
                                let prepared_commit = registry.prepare_commit(
                                    &job_id,
                                    stage_id,
                                    generation,
                                    task_id,
                                    committed.clone(),
                                )?;

                                // Keep the registry lock across the infallible
                                // graph transition and prepared metadata
                                // commit, so generation/version cannot change
                                // between validation and visibility.
                                running_stage
                                    .apply_task_info(task_id, task_status_for_update);
                                running_stage
                                    .apply_prepared_task_metrics(prepared_metrics);
                                running_stage
                                    .append_runtime_stats_reports(task_id, runtime_stats);
                                running_stage
                                    .append_window_state_reports(task_id, window_state);
                                registry.commit_prepared(prepared_commit);
                            } else {
                                // Feature off (or deliberately ineffective,
                                // e.g. AQE still enabled): preserve the
                                // pre-pipelining status/metrics/report ordering
                                // and behavior without touching the new
                                // registry mutex.
                                if !running_stage
                                    .update_task_info(task_id, task_status_for_update)
                                {
                                    continue;
                                }
                                running_stage
                                    .update_task_metrics(task_id, operator_metrics)?;
                                running_stage
                                    .append_runtime_stats_reports(task_id, runtime_stats);
                                running_stage
                                    .append_window_state_reports(task_id, window_state);
                            }
                            locations.append(&mut committed);
                        } else {
                            warn!(
                                "The task {task_identity}'s status is invalid for updating"
                            );
                        }
                    }

                    let is_final_successful = running_stage.is_successful()
                        && !lost_materialized_tasks.contains_key(&stage_id);
                    if is_final_successful {
                        if pipelining_enabled {
                            let mut registry = self.shuffle_inputs.lock();
                            let generation = registry
                                .get(&job_id, stage_id)
                                .map(|s| s.generation())
                                .ok_or_else(|| {
                                    BallistaError::Internal(format!(
                                        "pipelined shuffle input missing while sealing job {job_id}, stage {stage_id}"
                                    ))
                                })?;
                            registry.seal(&job_id, stage_id, generation)?;
                        }
                        successful_stages.insert(stage_id);
                        // if this stage is final successful, we want to combine the stage metrics to plan's metric set and print out the plan
                        if let Some(stage_metrics) = running_stage.stage_metrics.as_ref()
                        {
                            print_stage_metrics(
                                &job_id,
                                stage_id,
                                running_stage.plan.as_ref(),
                                stage_metrics,
                            );
                        }
                        log_merged_runtime_stats(
                            job_id.as_str(),
                            stage_id,
                            &running_stage.runtime_stats_reports,
                        );
                    }

                    let output_links = running_stage.output_links.clone();
                    resolved_stages.extend(
                        &mut self
                            .update_stage_output_links(
                                stage_id,
                                is_final_successful,
                                locations,
                                output_links,
                            )?
                            .into_iter(),
                    );
                } else if let ExecutionStage::UnResolved(unsolved_stage) = stage {
                    for task_status in stage_task_statuses.into_iter() {
                        let task_stage_attempt_num =
                            task_status.stage_attempt_num as usize;
                        let task_id = task_status.task_id as usize;
                        let task_identity = format!(
                            "TID {}/{}.{}/{}",
                            job_id, stage_id, task_stage_attempt_num, task_id
                        );
                        let mut should_ignore = true;
                        // handle delayed failed tasks if the stage's next attempt is still in UnResolved status.
                        if let Some(task_status::Status::Failed(failed_task)) =
                            task_status.status
                            && unsolved_stage.stage_attempt_num - task_stage_attempt_num
                                == 1
                        {
                            let failed_reason = failed_task.failed_reason;
                            match failed_reason {
                                Some(FailedReason::ExecutionError(_)) => {
                                    should_ignore = false;
                                    failed_stages.insert(stage_id, failed_task.error);
                                }
                                Some(FailedReason::FetchPartitionError(
                                    fetch_partiton_error,
                                )) if failed_stages.is_empty()
                                    && current_running_stages.contains(
                                        &(fetch_partiton_error.map_stage_id as usize),
                                    )
                                    && !unsolved_stage
                                        .last_attempt_failure_reasons
                                        .contains(&fetch_partiton_error.executor_id) =>
                                {
                                    should_ignore = false;
                                    unsolved_stage
                                        .last_attempt_failure_reasons
                                        .insert(fetch_partiton_error.executor_id.clone());
                                    let map_stage_id =
                                        fetch_partiton_error.map_stage_id as usize;
                                    let map_partition_id =
                                        fetch_partiton_error.map_partition_id as usize;
                                    let executor_id = fetch_partiton_error.executor_id;
                                    let removed_map_partitions = unsolved_stage
                                        .remove_input_partitions(
                                            map_stage_id,
                                            map_partition_id,
                                            &executor_id,
                                        )?;

                                    let lost_tasks = lost_materialized_tasks
                                        .entry(map_stage_id)
                                        .or_default();
                                    lost_tasks.extend(removed_map_partitions);
                                    warn!(
                                        "Need to reset the current running Stage {map_stage_id} due to late come FetchPartitionError from its parent stage {stage_id} of task {task_identity}"
                                    );

                                    // If the previous other task updates had already mark the map stage success, need to remove it.
                                    if successful_stages.contains(&map_stage_id) {
                                        successful_stages.remove(&map_stage_id);
                                    }
                                    if resolved_stages.contains(&stage_id) {
                                        resolved_stages.remove(&stage_id);
                                    }
                                }
                                _ => {}
                            }
                        }
                        if should_ignore {
                            warn!(
                                "Ignore TaskStatus update of task with TID {task_identity} as the Stage {job_id}/{stage_id} is in UnResolved status"
                            );
                        }
                    }
                } else {
                    warn!(
                        "Stage {}/{} is not in running when updating the status of tasks {:?}",
                        job_id,
                        stage_id,
                        stage_task_statuses
                            .into_iter()
                            .map(|task_status| task_status.task_id)
                            .collect::<Vec<_>>(),
                    );
                }
            } else {
                return Err(BallistaError::Internal(format!(
                    "Invalid stage ID {stage_id} for job {job_id}"
                )));
            }
        }

        // Update failed stage attempts back to self
        for (stage_id, attempts) in failed_stage_attempts.iter() {
            self.failed_stage_attempts
                .insert(*stage_id, HashSet::from_iter(attempts.iter().copied()));
        }

        // `remove_input_partitions` returns `loc.map_partition_id`, which
        // under the append-only multi-partition task model is the producing
        // task's task_id. A pipelined consumer can report a fetch failure while
        // that producer stage is still Running, so handle both producer states:
        //
        // * Running: retire the successful task in-place as ResultLost and put
        //   its whole input slice back in pending. Registry reconciliation then
        //   notices that an accepted publication disappeared and rolls the
        //   generation forward.
        // * Successful: mark the task ResultLost and use the existing
        //   Successful -> Running resubmission transition.
        //
        // Resetting the whole task slice is conservative but correct when one
        // artifact from a multi-partition task is lost.
        let mut successful_resubmits = HashSet::new();
        for (stage_id, lost_task_ids) in &lost_materialized_tasks {
            let Some(stage) = self.stages.get_mut(stage_id) else {
                return Err(BallistaError::Internal(format!(
                    "Invalid stage ID {stage_id} for job {job_id}"
                )));
            };

            match stage {
                ExecutionStage::Running(running_stage) => {
                    for task_id in lost_task_ids {
                        if *task_id >= running_stage.task_infos.len() {
                            return Err(BallistaError::Internal(format!(
                                "Invalid task_id {} in map stage {} (task_infos has {} entries)",
                                *task_id,
                                stage_id,
                                running_stage.task_infos.len()
                            )));
                        }
                        running_stage.mark_materialized_result_lost(
                            *task_id,
                            "FetchPartitionError in downstream stage",
                        );
                    }
                    // A producer may have been provisionally classified as
                    // successful earlier in this same status batch. Result
                    // loss wins and keeps it Running.
                    successful_stages.remove(stage_id);
                }
                ExecutionStage::Successful(success_stage) => {
                    for task_id in lost_task_ids {
                        if *task_id >= success_stage.task_infos.len() {
                            return Err(BallistaError::Internal(format!(
                                "Invalid task_id {} in map stage {} (task_infos has {} entries)",
                                *task_id,
                                stage_id,
                                success_stage.task_infos.len()
                            )));
                        }
                        let task_info = &mut success_stage.task_infos[*task_id];
                        if matches!(
                            task_info.task_status,
                            task_status::Status::Successful(_)
                        ) {
                            task_info.task_status =
                                task_status::Status::Failed(FailedTask {
                                    error: "FetchPartitionError in downstream stage"
                                        .to_owned(),
                                    retryable: true,
                                    count_to_failures: false,
                                    failed_reason: Some(FailedReason::ResultLost(
                                        ResultLost {},
                                    )),
                                });
                        }
                    }
                    successful_resubmits.insert(*stage_id);
                }
                _ => {
                    warn!(
                        "Ignoring materialized-result loss for non-running/non-successful stage {job_id}/{stage_id}"
                    );
                }
            }

            // If this producer was sealed earlier in the same batch, any
            // dependent stage that was tentatively marked resolvable must not
            // cross the barrier using the now-lost output.
            let output_links = stage.output_links().to_vec();
            for output_stage_id in output_links {
                resolved_stages.remove(&output_stage_id);
            }
        }

        let mut events = self.processing_stages_update(UpdatedStages {
            resolved_stages,
            successful_stages,
            failed_stages,
            rollback_running_stages,
            resubmit_successful_stages: successful_resubmits,
        })?;
        let deferred = std::mem::take(&mut self.deferred_successes);
        let mut ready = vec![];
        for (executor, status) in deferred {
            let Some(ExecutionStage::Running(stage)) =
                self.stages.get(&(status.stage_id as usize))
            else {
                continue;
            };
            if stage.stage_attempt_num != status.stage_attempt_num as usize {
                continue;
            }
            let complete = match &stage.admission {
                super::execution_stage::StageAdmission::Normal => true,
                super::execution_stage::StageAdmission::TailPipelined(handles) => {
                    self.input_histories_ready(handles, true)
                }
            };
            if complete {
                ready.push((executor, status));
            } else {
                self.deferred_successes.push((executor, status));
            }
        }
        for (executor, status) in ready {
            events.extend(self.update_task_status(
                &executor,
                vec![status],
                max_task_failures,
                max_stage_failures,
            )?);
        }
        Ok(events)
    }

    /// Return all the currently running stage ids
    fn running_stages(&self) -> Vec<usize> {
        self.stages
            .iter()
            .filter_map(|(stage_id, stage)| {
                if let ExecutionStage::Running(_running) = stage {
                    Some(*stage_id)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    }

    /// Return all currently running tasks along with the executor ID on which they are assigned
    fn running_tasks(&self) -> Vec<RunningTaskInfo> {
        self.stages
            .values()
            .flat_map(|stage| {
                if let ExecutionStage::Running(stage) = stage {
                    stage
                        .running_tasks()
                        .into_iter()
                        .map(|(task_id, stage_id, executor_id)| RunningTaskInfo {
                            task_id,
                            job_id: self.job_id.clone(),
                            stage_id,
                            executor_id,
                        })
                        .collect::<Vec<RunningTaskInfo>>()
                } else {
                    vec![]
                }
            })
            .collect::<Vec<RunningTaskInfo>>()
    }

    /// Total number of tasks in this plan that are ready for scheduling
    fn available_tasks(&self) -> usize {
        self.stages
            .values()
            .map(|stage| {
                if let ExecutionStage::Running(stage) = stage {
                    stage.available_tasks()
                } else {
                    0
                }
            })
            .sum()
    }

    fn fetch_running_stage(&mut self, black_list: &[usize]) -> Option<&mut RunningStage> {
        if matches!(
            self.status,
            JobStatus {
                status: Some(job_status::Status::Failed(_)),
                ..
            }
        ) {
            debug!("Call fetch_runnable_stage on failed Job");
            return None;
        }

        // Preserve the legacy API as a safe normal-work-only interface when
        // pipelining is enabled. Existing custom distribution policies may call
        // fetch_running_stage directly; exposing a TailPipelined stage here
        // would let them bypass global normal-first admission and the producer
        // readiness checks. Such policies keep working conservatively (without
        // overlap) until they explicitly adopt fetch_running_stage_for_admission.
        if self.pipelining_enabled() {
            let mut selected = self.running_stage_for_admission_id(black_list, false);
            if selected.is_none() && self.revive() {
                selected = self.running_stage_for_admission_id(black_list, false);
            }
            return match selected.and_then(|id| self.stages.get_mut(&id)) {
                Some(ExecutionStage::Running(stage)) => Some(stage),
                _ => None,
            };
        }

        let running_stage_id = self.get_running_stage_id(black_list);
        if let Some(running_stage_id) = running_stage_id {
            if let Some(ExecutionStage::Running(running_stage)) =
                self.stages.get_mut(&running_stage_id)
            {
                Some(running_stage)
            } else {
                warn!("Fail to find running stage with id {running_stage_id}");
                None
            }
        } else {
            None
        }
    }

    fn fetch_running_stage_for_admission(
        &mut self,
        black_list: &[usize],
        tail: bool,
    ) -> Option<&mut RunningStage> {
        if !self.pipelining_enabled() {
            return if tail {
                None
            } else {
                self.fetch_running_stage(black_list)
            };
        }
        if !matches!(self.status.status, Some(job_status::Status::Running(_))) {
            return None;
        }
        if tail
            && self.pipelined_resolution_error.is_none()
            && let Err(err) = self.resolve_tail_stages()
        {
            let message = err.to_string();
            error!(
                "Job {}: deterministic pipelined shuffle plan construction failed; \
                 disabling further tail plan construction for this job and falling back to the blocking barrier: {}",
                self.job_id, message
            );
            self.pipelined_resolution_error = Some(message);
            return None;
        }

        let mut selected = self.running_stage_for_admission_id(black_list, tail);

        // Preserve the legacy scheduler's liveness transition: a fresh graph
        // starts with source stages in Resolved state, and normal scheduling is
        // responsible for reviving them before the first task can be bound.
        // Tail admission never revives stages on its own.
        if selected.is_none() && !tail && self.revive() {
            selected = self.running_stage_for_admission_id(black_list, false);
        }

        match selected.and_then(|id| self.stages.get_mut(&id)) {
            Some(ExecutionStage::Running(stage)) => Some(stage),
            _ => None,
        }
    }

    fn update_status(&mut self, status: JobStatus) {
        self.status = status;
    }

    fn output_locations(&self) -> Vec<PartitionLocation> {
        self.output_locations.clone()
    }

    /// Reset running and successful stages on a given executor
    /// This will first check the unresolved/resolved/running stages and reset the running tasks and successful tasks.
    /// Then it will check the successful stage and whether there are running parent stages need to read shuffle from it.
    /// If yes, reset the successful tasks and roll back the resolved shuffle recursively.
    ///
    /// Returns the reset stage ids and running tasks should be killed
    fn reset_stages_on_lost_executor(
        &mut self,
        executor_id: &str,
    ) -> Result<(HashSet<usize>, Vec<RunningTaskInfo>)> {
        let mut reset = HashSet::new();
        let mut tasks_to_cancel = vec![];
        loop {
            let reset_stage = self.reset_stages_internal(executor_id)?;
            if !reset_stage.0.is_empty() {
                reset.extend(reset_stage.0.iter());
                tasks_to_cancel.extend(reset_stage.1)
            } else {
                tasks_to_cancel.extend(self.reconcile_pipelined()?);
                return Ok((reset, tasks_to_cancel));
            }
        }
    }

    /// Convert unresolved stage to be resolved
    fn resolve_stage(&mut self, stage_id: usize) -> Result<bool> {
        if let Some(ExecutionStage::UnResolved(stage)) = self.stages.get(&stage_id) {
            let resolved_stage = stage.to_resolved(self.session_config.options())?;
            self.stages
                .insert(stage_id, ExecutionStage::Resolved(resolved_stage));
            Ok(true)
        } else {
            warn!(
                "Fail to find a unresolved stage {}/{} to resolve",
                self.job_id(),
                stage_id
            );
            Ok(false)
        }
    }

    /// Convert running stage to be successful
    fn succeed_stage(&mut self, stage_id: usize) -> bool {
        if let Some(ExecutionStage::Running(stage)) = self.stages.remove(&stage_id) {
            self.stages
                .insert(stage_id, ExecutionStage::Successful(stage.to_successful()));
            self.clear_stage_failure(stage_id);
            true
        } else {
            warn!(
                "Fail to find a running stage {}/{} to make it success",
                self.job_id(),
                stage_id
            );
            false
        }
    }

    /// Convert running stage to be failed
    fn fail_stage(&mut self, stage_id: usize, err_msg: String) -> bool {
        if let Some(ExecutionStage::Running(stage)) = self.stages.remove(&stage_id) {
            self.stages
                .insert(stage_id, ExecutionStage::Failed(stage.to_failed(err_msg)));
            true
        } else {
            info!(
                "Fail to find a running stage {}/{} to fail",
                self.job_id(),
                stage_id
            );
            false
        }
    }

    /// Convert running stage to be unresolved,
    /// Returns a Vec of RunningTaskInfo for running tasks in this stage.
    fn rollback_running_stage(
        &mut self,
        stage_id: usize,
        failure_reasons: HashSet<String>,
    ) -> Result<Vec<RunningTaskInfo>> {
        if let Some(ExecutionStage::Running(stage)) = self.stages.get(&stage_id) {
            let running_tasks = stage
                .running_tasks()
                .into_iter()
                .map(|(task_id, stage_id, executor_id)| RunningTaskInfo {
                    task_id,
                    job_id: self.job_id.clone(),
                    stage_id,
                    executor_id,
                })
                .collect();
            let unresolved_stage = stage.to_unresolved(failure_reasons)?;
            if self.pipelining_enabled() {
                // task_infos is append-only across attempts. A restored prefix
                // belongs to older attempts and exists only to reserve task IDs;
                // never manufacture (new_attempt, old_task_id) refund identities.
                let current_attempt_start =
                    self.retired_tasks.get(&stage_id).map(Vec::len).unwrap_or(0);
                let mut retired = stage.task_infos.clone();
                for (task, info) in retired.iter_mut().enumerate() {
                    let key = (stage_id, stage.stage_attempt_num, task);
                    if task >= current_attempt_start
                        && !self.refunded_tasks.contains(&key)
                        && let task_status::Status::Running(owner) = &info.task_status
                    {
                        self.retired_vcores.insert(
                            key,
                            (owner.executor_id.clone(), info.vcores_consumed),
                        );
                    }
                    info.task_status = task_status::Status::Failed(FailedTask {
                        error: "tail attempt retired".into(),
                        retryable: false,
                        count_to_failures: false,
                        failed_reason: Some(FailedReason::TaskKilled(
                            ballista_core::serde::protobuf::TaskKilled {},
                        )),
                    });
                }
                self.retired_tasks.insert(stage_id, retired);
            }
            self.stages
                .insert(stage_id, ExecutionStage::UnResolved(unresolved_stage));
            Ok(running_tasks)
        } else {
            warn!(
                "Fail to find a running stage {}/{} to rollback",
                self.job_id(),
                stage_id
            );
            Ok(vec![])
        }
    }

    /// Convert resolved stage to be unresolved
    fn rollback_resolved_stage(&mut self, stage_id: usize) -> Result<bool> {
        if let Some(ExecutionStage::Resolved(stage)) = self.stages.get(&stage_id) {
            let unresolved_stage = stage.to_unresolved()?;
            self.stages
                .insert(stage_id, ExecutionStage::UnResolved(unresolved_stage));
            Ok(true)
        } else {
            warn!(
                "Fail to find a resolved stage {}/{} to rollback",
                self.job_id(),
                stage_id
            );
            Ok(false)
        }
    }

    /// Convert successful stage to be running
    fn rerun_successful_stage(&mut self, stage_id: usize) -> bool {
        if let Some(ExecutionStage::Successful(stage)) = self.stages.remove(&stage_id) {
            self.stages
                .insert(stage_id, ExecutionStage::Running(stage.to_running()));
            true
        } else {
            warn!(
                "Fail to find a successful stage {}/{} to rerun",
                self.job_id(),
                stage_id
            );
            false
        }
    }

    /// fail job with error message
    fn fail_job(&mut self, error: String) {
        self.end_time = timestamp_millis();

        self.status = JobStatus {
            job_id: self.job_id.clone().into(),
            job_name: self.job_name.clone(),
            status: Some(Status::Failed(FailedJob {
                error,
                queued_at: self.queued_at,
                started_at: self.start_time,
                ended_at: self.end_time,
            })),
        };
    }

    /// Mark the job success
    fn succeed_job(&mut self) -> Result<()> {
        if !self.is_successful() {
            return Err(BallistaError::Internal(format!(
                "Attempt to finalize an incomplete job {}",
                self.job_id()
            )));
        }

        let partition_location = self
            .output_locations()
            .into_iter()
            .map(|l| l.try_into())
            .collect::<Result<Vec<_>>>()?;

        self.end_time = timestamp_millis();

        self.status = JobStatus {
            job_id: self.job_id.clone().into(),
            job_name: self.job_name.clone(),
            status: Some(job_status::Status::Successful(SuccessfulJob {
                partition_location,

                queued_at: self.queued_at,
                started_at: self.start_time,
                ended_at: self.end_time,
            })),
        };

        Ok(())
    }

    fn stages(&self) -> &HashMap<usize, ExecutionStage> {
        &self.stages
    }

    fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Get next task that can be assigned to the given executor.
    /// This method should only be called when the resulting task is immediately
    /// being launched as the status will be set to Running and it will not be
    /// available to the scheduler.
    /// If the task is not launched the status must be reset to allow the task to
    /// be scheduled elsewhere.
    #[cfg(test)]
    fn pop_next_task(&mut self, executor_id: &str) -> Result<Option<TaskDescription>> {
        if matches!(
            self.status,
            JobStatus {
                status: Some(job_status::Status::Failed(_)),
                ..
            }
        ) {
            warn!("Call pop_next_task on failed Job");
            return Ok(None);
        }

        let job_id = self.job_id.clone();
        let session_id = self.session_id.clone();

        let mut next_task = self.stages.iter_mut().find(|(_stage_id, stage)| {
            if let ExecutionStage::Running(stage) = stage {
                stage.available_tasks() > 0
            } else {
                false
            }
        }).map(|(stage_id, stage)| {
            if let ExecutionStage::Running(stage) = stage {
                // pop_next_task hands out a single-partition task — bind path
                // sized to `exec.vcores` lives in `cluster::bind_task_*`.
                let input_partition_ids = stage.pending.next_slice(1);
                if input_partition_ids.is_empty() {
                    return Err(BallistaError::Internal(format!(
                        "Error getting next task for job {job_id}: Stage {stage_id} is ready but has no pending tasks"
                    )));
                }
                // task_id is the append slot in `task_infos` — assigned as
                // `task_infos.len()` at bind time. `(job_id, stage_id, task_id)`
                // is globally unique.
                let task_id = stage.task_infos.len();
                let task_attempt = input_partition_ids
                    .iter()
                    .map(|pid| stage.task_failure_numbers[*pid])
                    .max()
                    .unwrap_or(0);
                let task_info = TaskInfo {
                    task_id,
                    scheduled_time: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis(),
                    // Those times will be updated when the task finish
                    launch_time: 0,
                    start_exec_time: 0,
                    end_exec_time: 0,
                    finish_time: 0,
                    task_status: task_status::Status::Running(RunningTask {
                        executor_id: executor_id.to_owned()
                    }),
                    global_input_partition_ids: input_partition_ids.clone(),
                    vcores_consumed: input_partition_ids.len() as u32,
                };
                stage.task_infos.push(task_info);

                let key = TaskKey {
                    job_id,
                    stage_id: *stage_id,
                    task_id,
                };

                let vcores_consumed = input_partition_ids.len() as u32;
                Ok(TaskDescription {
                    session_id,
                    key,
                    stage_attempt_num: stage.stage_attempt_num,
                    task_attempt,
                    global_input_partition_ids: input_partition_ids,
                    vcores_consumed,
                    plan: stage.plan.clone(),
                    session_config: self.session_config.clone()
                })
            } else {
                Err(BallistaError::General(format!("Stage {stage_id} is not a running stage")))
            }
        }).transpose()?;

        // If no available tasks found in the running stage,
        // try to find a resolved stage and convert it to the running stage
        if next_task.is_none() {
            if self.revive() {
                next_task = self.pop_next_task(executor_id)?;
            } else {
                next_task = None;
            }
        }

        Ok(next_task)
    }
}

impl Debug for StaticExecutionGraph {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let stages = self
            .stages
            .values()
            .map(|stage| format!("{stage:?}"))
            .collect::<Vec<String>>()
            .join("");
        write!(
            f,
            "ExecutionGraph[job_id={}, session_id={}, available_tasks={}, is_successful={}]\n{}",
            self.job_id,
            self.session_id,
            self.available_tasks(),
            self.is_successful(),
            stages
        )
    }
}

/// Creates a new `TaskInfo` for a task that is about to be scheduled on an
/// executor. The caller sets `global_input_partition_ids` to the partitions this task
/// will process (bind loops draw the slice from `stage.pending`).
pub fn create_task_info(executor_id: String, task_id: usize) -> TaskInfo {
    TaskInfo {
        task_id,
        scheduled_time: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis(),
        launch_time: 0,
        start_exec_time: 0,
        end_exec_time: 0,
        finish_time: 0,
        task_status: task_status::Status::Running(RunningTask { executor_id }),
        global_input_partition_ids: vec![],
        vcores_consumed: 0,
    }
}

/// Utility for building a set of `ExecutionStage`s from
/// a list of `ShuffleWriterExec`.
///
/// This will infer the dependency structure for the stages
/// so that we can construct a DAG from the stages.
pub(crate) struct ExecutionStageBuilder {
    /// Stage ID which is currently being visited
    current_stage_id: usize,
    /// Map from stage ID -> List of child stage IDs
    stage_dependencies: HashMap<usize, Vec<usize>>,
    /// Map from Stage ID -> output link
    output_links: HashMap<usize, Vec<usize>>,
    session_config: Arc<SessionConfig>,
}

impl ExecutionStageBuilder {
    pub fn new(session_config: Arc<SessionConfig>) -> Self {
        Self {
            current_stage_id: 0,
            stage_dependencies: HashMap::new(),
            output_links: HashMap::new(),
            session_config,
        }
    }

    pub fn build(
        mut self,
        stages: Vec<Arc<dyn ShuffleWriter>>,
    ) -> Result<HashMap<usize, ExecutionStage>> {
        let mut execution_stages: HashMap<usize, ExecutionStage> = HashMap::new();
        // First, build the dependency graph
        for stage in &stages {
            accept(stage.as_ref(), &mut self)?;
        }

        // Now, create the execution stages
        for stage in stages {
            let stage_id = stage.stage_id();
            let output_links = self.output_links.remove(&stage_id).unwrap_or_default();

            let child_stages = self
                .stage_dependencies
                .remove(&stage_id)
                .unwrap_or_default();

            let stage = if child_stages.is_empty() {
                ExecutionStage::Resolved(ResolvedStage::new(
                    stage_id,
                    0,
                    stage,
                    output_links,
                    HashMap::new(),
                    HashSet::new(),
                    self.session_config.clone(),
                ))
            } else {
                ExecutionStage::UnResolved(UnresolvedStage::new(
                    stage_id,
                    stage,
                    output_links,
                    child_stages,
                    self.session_config.clone(),
                ))
            };
            execution_stages.insert(stage_id, stage);
        }

        Ok(execution_stages)
    }
}

impl ExecutionPlanVisitor for ExecutionStageBuilder {
    type Error = BallistaError;

    fn pre_visit(
        &mut self,
        plan: &dyn ExecutionPlan,
    ) -> std::result::Result<bool, Self::Error> {
        // Handle both ShuffleWriterExec and SortShuffleWriterExec
        if let Some(shuffle_write) = plan.downcast_ref::<ShuffleWriterExec>() {
            self.current_stage_id = shuffle_write.stage_id();
        } else if let Some(shuffle_write) = plan.downcast_ref::<RangeShuffleWriterExec>()
        {
            self.current_stage_id = shuffle_write.stage_id();
        } else if let Some(shuffle_write) = plan.downcast_ref::<SortShuffleWriterExec>() {
            self.current_stage_id = shuffle_write.stage_id();
        } else if let Some(unresolved_shuffle) =
            plan.downcast_ref::<UnresolvedShuffleExec>()
        {
            if let Some(output_links) =
                self.output_links.get_mut(&unresolved_shuffle.stage_id)
            {
                if !output_links.contains(&self.current_stage_id) {
                    output_links.push(self.current_stage_id);
                }
            } else {
                self.output_links
                    .insert(unresolved_shuffle.stage_id, vec![self.current_stage_id]);
            }

            if let Some(deps) = self.stage_dependencies.get_mut(&self.current_stage_id) {
                if !deps.contains(&unresolved_shuffle.stage_id) {
                    deps.push(unresolved_shuffle.stage_id);
                }
            } else {
                self.stage_dependencies
                    .insert(self.current_stage_id, vec![unresolved_shuffle.stage_id]);
            }
        }
        Ok(true)
    }
}

/// Represents the basic unit of work for the Ballista executor.
///
/// One `TaskDescription` drives all of `global_input_partition_ids`'s partitions
/// through one plan-Arc on the assigned executor.
#[derive(Clone)]
pub struct TaskDescription {
    /// The session ID associated with this task's job.
    pub session_id: String,
    /// Task locator: `(job_id, stage_id, task_id)`. `task_id` is this task's
    /// append-order slot in `RunningStage.task_infos`.
    pub key: TaskKey,
    /// The attempt number for this stage (for retry tracking).
    pub stage_attempt_num: usize,
    /// The attempt number for this specific task (for retry tracking).
    pub task_attempt: usize,
    /// The partitions (real plan input indices) this task will process.
    /// Populated at bind time from the stage's `PendingPartitions` cursor
    /// sized to the assigned executor's free vcores. Baked into `plan`
    /// via `task_builder::restrict_plan_to_partitions` before dispatch.
    pub global_input_partition_ids: Vec<usize>,
    /// Vcores this task consumed from the executor's budget at bind time
    /// (`min(global_input_partition_ids.len(), budget.vcores)` for non-collapse
    /// stages, `1` for collapse stages). Forwarded to the executor over the
    /// wire so the memory pool can be sized proportionally.
    pub vcores_consumed: u32,
    /// The physical execution plan to run for this task.
    pub plan: Arc<dyn ExecutionPlan>,
    /// Session configuration for this task's execution context.
    pub session_config: Arc<SessionConfig>,
}

impl Debug for TaskDescription {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let plan = DisplayableExecutionPlan::new(self.plan.as_ref()).indent(false);
        write!(
            f,
            "TaskDescription[session_id: {},job: {}, stage: {}.{}, task_id: {}, task attempt {}]\n{}",
            self.session_id,
            self.key.job_id,
            self.key.stage_id,
            self.stage_attempt_num,
            self.key.task_id,
            self.task_attempt,
            plan
        )
    }
}

impl TaskDescription {
    /// Returns the number of output partitions this task will produce.
    pub fn get_output_partition_number(&self) -> usize {
        // Try ShuffleWriterExec first
        if let Some(shuffle_writer) = self.plan.downcast_ref::<ShuffleWriterExec>() {
            return shuffle_writer
                .shuffle_output_partitioning()
                .map(|partitioning| partitioning.partition_count())
                .unwrap_or(1);
        }
        // Try SortShuffleWriterExec
        if let Some(shuffle_writer) = self.plan.downcast_ref::<SortShuffleWriterExec>() {
            return shuffle_writer
                .shuffle_output_partitioning()
                .partition_count();
        }
        // Default fallback
        1
    }
}

pub(crate) fn partition_to_location(
    job_id: &JobId,
    map_partition_id: usize,
    stage_id: usize,
    executor: &ExecutorMetadata,
    shuffles: Vec<ShuffleWritePartition>,
) -> Vec<PartitionLocation> {
    shuffles
        .into_iter()
        .map(|shuffle| PartitionLocation {
            map_partition_id,
            partition_id: PartitionId {
                job_id: job_id.to_owned(),
                stage_id,
                partition_id: shuffle.partition_id as usize,
            },
            executor_meta: executor.clone(),
            partition_stats: PartitionStats::new(
                Some(shuffle.num_rows),
                Some(shuffle.num_batches),
                Some(shuffle.num_bytes),
            ),
            file_id: shuffle.file_id,
            is_sort_shuffle: shuffle.is_sort_shuffle,
        })
        .collect()
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;
    use std::sync::Arc;

    use crate::scheduler_server::event::QueryStageSchedulerEvent;
    use ballista_core::JobId;
    use ballista_core::error::{BallistaError, Result};
    use ballista_core::execution_plans::UnresolvedShuffleExec;
    use ballista_core::extension::SessionConfigExt;
    use ballista_core::serde::protobuf::{
        self, ExecutionError, FailedTask, FetchPartitionError, IoError, JobStatus,
        TaskKilled, failed_task, job_status, task_status,
    };
    use datafusion::common::tree_node::TreeNodeRecursion;
    use datafusion::common::{DataFusionError, Result as DataFusionResult};
    use datafusion::execution::TaskContext;
    use datafusion::physical_expr::PhysicalExpr;
    use datafusion::physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
        SendableRecordBatchStream,
    };
    use datafusion::prelude::SessionConfig;

    use crate::state::execution_graph::ExecutionGraph;
    use crate::state::execution_stage::ExecutionStage;
    use crate::test_utils::{
        mock_completed_task, mock_executor, mock_failed_task,
        revive_graph_and_complete_next_stage,
        revive_graph_and_complete_next_stage_with_executor, test_aggregation_plan,
        test_aggregation_plan_with_config, test_coalesce_plan, test_join_plan,
        test_two_aggregations_plan, test_union_all_plan, test_union_plan,
    };

    async fn pipelined_fresh_graph() -> super::StaticExecutionGraph {
        let mut graph = test_two_aggregations_plan(4).await;
        graph.session_config = Arc::new(
            graph
                .session_config
                .as_ref()
                .clone()
                .with_ballista_adaptive_query_planner(false)
                .with_ballista_shuffle_pipelined_enabled(true),
        );
        for id in graph.stages.keys() {
            graph.shuffle_inputs.lock().create(&graph.job_id, *id);
        }
        graph
    }

    async fn pipelined_test_graph() -> super::StaticExecutionGraph {
        let mut graph = pipelined_fresh_graph().await;
        graph.revive();
        graph
    }

    fn bind_tail(
        graph: &mut super::StaticExecutionGraph,
        executor: &str,
    ) -> super::TaskDescription {
        let job_id = graph.job_id.clone();
        let session_id = graph.session_id.clone();
        let stage = graph
            .fetch_running_stage_for_admission(&[], true)
            .expect("tail should be admitted");
        let partitions = stage.pending.next_slice(1);
        let task_id = stage.task_infos.len();
        let mut info = super::create_task_info(executor.into(), task_id);
        info.global_input_partition_ids = partitions.clone();
        info.vcores_consumed = 1;
        stage.task_infos.push(info);
        super::TaskDescription {
            session_id,
            key: ballista_core::serde::scheduler::TaskKey {
                job_id,
                stage_id: stage.stage_id,
                task_id,
            },
            stage_attempt_num: stage.stage_attempt_num,
            task_attempt: 0,
            global_input_partition_ids: partitions,
            vcores_consumed: 1,
            plan: stage.plan.clone(),
            session_config: stage.session_config.clone(),
        }
    }

    #[tokio::test]
    async fn legacy_running_stage_api_never_exposes_tail_work() -> Result<()> {
        let mut graph = pipelined_test_graph().await;
        let executor = mock_executor("producer".into());

        // Publish one producer task, then assign every remaining producer
        // partition without completing it. That makes the intermediate
        // consumer tail-eligible: committed history exists and producer
        // pending is empty, but the producer has not sealed.
        let first = graph.pop_next_task(&executor.id)?.unwrap();
        graph.update_task_status(
            &executor,
            vec![mock_completed_task(first, &executor.id)],
            4,
            4,
        )?;
        loop {
            let Some(stage) = graph.fetch_running_stage_for_admission(&[], false) else {
                break;
            };
            let partitions = stage.pending.next_slice(1);
            if partitions.is_empty() {
                break;
            }
            let task_id = stage.task_infos.len();
            let mut info =
                super::create_task_info("producer-straggler".into(), task_id);
            info.global_input_partition_ids = partitions;
            info.vcores_consumed = 1;
            stage.task_infos.push(info);
        }

        assert!(
            graph.fetch_running_stage(&[]).is_none(),
            "legacy custom-policy API must not expose revocable tail work"
        );
        assert!(
            graph.fetch_running_stage_for_admission(&[], true).is_some(),
            "admission-aware API should expose the eligible tail"
        );
        Ok(())
    }

    #[tokio::test]
    async fn pipelined_normal_admission_revives_a_fresh_graph() -> Result<()> {
        let mut graph = pipelined_fresh_graph().await;
        assert!(graph.running_stages().is_empty());

        let stage = graph
            .fetch_running_stage_for_admission(&[], false)
            .expect("fresh graph must revive normal producer work");
        assert!(matches!(
            &stage.admission,
            super::super::execution_stage::StageAdmission::Normal
        ));
        Ok(())
    }

    #[tokio::test]
    async fn pipelined_reports_require_live_task_ownership() -> Result<()> {
        let mut graph = pipelined_test_graph().await;
        let executor = mock_executor("owner".into());
        let task = graph.pop_next_task(&executor.id)?.unwrap();
        let stage_id = task.key.stage_id;
        let success = mock_completed_task(task, &executor.id);

        let wrong_executor = mock_executor("another-executor".into());
        assert_eq!(graph.release_task_vcores(&wrong_executor.id, &success), 0);
        graph.update_task_status(&wrong_executor, vec![success.clone()], 4, 4)?;

        let mut malformed = success.clone();
        malformed.status = None;
        graph.update_task_status(&executor, vec![malformed], 4, 4)?;
        let mut wrong_payload = success.clone();
        if let Some(task_status::Status::Successful(ref mut task)) = wrong_payload.status
        {
            task.executor_id = wrong_executor.id.clone();
        }
        assert_eq!(graph.release_task_vcores(&executor.id, &wrong_payload), 0);
        graph.update_task_status(&executor, vec![wrong_payload], 4, 4)?;
        assert!(
            !graph
                .shuffle_inputs
                .lock()
                .get(&graph.job_id, stage_id)
                .unwrap()
                .has_committed_input()
        );

        assert_eq!(graph.release_task_vcores(&executor.id, &success), 1);
        graph.update_task_status(&executor, vec![success.clone()], 4, 4)?;
        assert!(
            graph
                .shuffle_inputs
                .lock()
                .get(&graph.job_id, stage_id)
                .unwrap()
                .has_committed_input()
        );

        let mut late_failure = success;
        late_failure.status = Some(task_status::Status::Failed(FailedTask {
            error: "late duplicate".into(),
            retryable: false,
            count_to_failures: true,
            failed_reason: Some(failed_task::FailedReason::ExecutionError(
                ExecutionError {},
            )),
        }));
        assert_eq!(graph.release_task_vcores(&executor.id, &late_failure), 0);
        graph.update_task_status(&executor, vec![late_failure], 4, 4)?;
        assert!(!matches!(
            graph.status().status,
            Some(job_status::Status::Failed(_))
        ));
        Ok(())
    }

    #[tokio::test]
    async fn pipelined_duplicate_failures_reschedule_partitions_once() -> Result<()> {
        let mut graph = pipelined_test_graph().await;
        let executor = mock_executor("owner".into());
        let task = graph.pop_next_task(&executor.id)?.unwrap();
        let stage_id = task.key.stage_id;
        let task_partitions = task.global_input_partition_ids.clone();
        let pending = graph.stages[&stage_id].task_infos().unwrap().len();
        assert_eq!(pending, 1);
        let failure = mock_failed_task(
            task,
            FailedTask {
                error: "retry once".into(),
                retryable: true,
                count_to_failures: true,
                failed_reason: Some(failed_task::FailedReason::IoError(IoError {})),
            },
        );
        graph.update_task_status(
            &executor,
            vec![failure.clone(), failure.clone()],
            4,
            4,
        )?;
        graph.update_task_status(&executor, vec![failure], 4, 4)?;

        let ExecutionStage::Running(stage) = &graph.stages[&stage_id] else {
            panic!("producer remains runnable after retryable failure");
        };
        assert_eq!(stage.pending.remaining(), stage.pending.total_partitions());
        for partition in task_partitions {
            assert_eq!(stage.task_failure_numbers[partition], 1);
        }
        Ok(())
    }

    #[tokio::test]
    async fn pipelined_repeated_rollbacks_preserve_task_ids_and_refund_once() -> Result<()>
    {
        let mut graph = pipelined_test_graph().await;

        let first = graph.pop_next_task("attempt-0")?.unwrap();
        let stage_id = first.key.stage_id;
        let first_task_id = first.key.task_id;
        let first_attempt = first.stage_attempt_num;
        let first_status = mock_completed_task(first, "attempt-0");

        graph.rollback_running_stage(stage_id, HashSet::new())?;
        assert_eq!(
            graph.release_task_vcores("wrong-executor", &first_status),
            0
        );
        assert_eq!(graph.release_task_vcores("attempt-0", &first_status), 1);
        assert_eq!(graph.release_task_vcores("attempt-0", &first_status), 0);

        graph.resolve_stage(stage_id)?;
        graph.revive();
        let second = graph.pop_next_task("attempt-1")?.unwrap();
        assert_eq!(second.key.stage_id, stage_id);
        assert!(second.key.task_id > first_task_id);
        assert!(second.stage_attempt_num > first_attempt);
        let second_task_id = second.key.task_id;
        let second_attempt = second.stage_attempt_num;
        let second_status = mock_completed_task(second, "attempt-1");

        graph.rollback_running_stage(stage_id, HashSet::new())?;
        assert_eq!(graph.release_task_vcores("attempt-1", &second_status), 1);
        assert_eq!(graph.release_task_vcores("attempt-1", &second_status), 0);
        // A very late duplicate from attempt N remains refunded even after N+1
        // has itself been retired.
        assert_eq!(graph.release_task_vcores("attempt-0", &first_status), 0);
        // Old append-only slots are identity reservations, not tasks from the
        // newer attempt. An impossible N+1/task-from-N status must not acquire
        // a synthetic refund entry during the second rollback.
        let mut impossible_identity = first_status.clone();
        impossible_identity.stage_attempt_num = second_attempt as u32;
        assert_eq!(
            graph.release_task_vcores("attempt-0", &impossible_identity),
            0
        );
        assert!(!graph.retired_vcores.contains_key(&(
            stage_id,
            second_attempt,
            first_task_id
        )));

        graph.resolve_stage(stage_id)?;
        graph.revive();
        let third = graph.pop_next_task("attempt-2")?.unwrap();
        assert_eq!(third.key.stage_id, stage_id);
        assert!(third.key.task_id > second_task_id);
        assert!(third.stage_attempt_num > second_attempt);
        Ok(())
    }

    #[tokio::test]
    async fn pipelined_stale_deferred_success_after_executor_loss_is_ignored()
    -> Result<()> {
        let mut graph = pipelined_test_graph().await;
        let producer_executor = mock_executor("producer".into());
        let first = graph.pop_next_task(&producer_executor.id)?.unwrap();
        graph.update_task_status(
            &producer_executor,
            vec![mock_completed_task(first, &producer_executor.id)],
            4,
            4,
        )?;
        let straggler = graph.pop_next_task(&producer_executor.id)?.unwrap();

        let consumer_executor = mock_executor("consumer".into());
        let consumer = bind_tail(&mut graph, &consumer_executor.id);
        let consumer_stage_id = consumer.key.stage_id;
        let consumer_task_id = consumer.key.task_id;
        let stale_success = mock_completed_task(consumer, &consumer_executor.id);

        graph.update_task_status(
            &consumer_executor,
            vec![stale_success.clone()],
            4,
            4,
        )?;
        assert_eq!(graph.deferred_successes.len(), 1);

        graph.reset_stages_on_lost_executor(&consumer_executor.id)?;
        assert!(matches!(
            &graph.stages[&consumer_stage_id].task_infos().unwrap()[consumer_task_id]
                .task_status,
            task_status::Status::Failed(FailedTask {
                failed_reason: Some(failed_task::FailedReason::ResultLost(_)),
                ..
            })
        ));

        // Sealing the producer replays the deferred status internally. The old
        // consumer attempt was reset, so that replay must not publish output.
        graph.update_task_status(
            &producer_executor,
            vec![mock_completed_task(straggler, &producer_executor.id)],
            4,
            4,
        )?;
        assert!(graph.deferred_successes.is_empty());
        assert!(
            !graph
                .shuffle_inputs
                .lock()
                .get(&graph.job_id, consumer_stage_id)
                .unwrap()
                .has_committed_input()
        );

        // A duplicate terminal report from the retired executor after seal is
        // equally stale and must remain harmless.
        graph.update_task_status(&consumer_executor, vec![stale_success], 4, 4)?;
        assert!(
            !graph
                .shuffle_inputs
                .lock()
                .get(&graph.job_id, consumer_stage_id)
                .unwrap()
                .has_committed_input()
        );

        let retry = graph.pop_next_task("consumer-retry")?.unwrap();
        assert_eq!(retry.key.stage_id, consumer_stage_id);
        assert!(retry.key.task_id > consumer_task_id);
        let retry_executor = mock_executor("consumer-retry".into());
        graph.update_task_status(
            &retry_executor,
            vec![mock_completed_task(retry, &retry_executor.id)],
            4,
            4,
        )?;
        assert!(
            graph
                .shuffle_inputs
                .lock()
                .get(&graph.job_id, consumer_stage_id)
                .unwrap()
                .has_committed_input()
        );
        Ok(())
    }

    #[tokio::test]
    async fn pipelined_tail_waits_for_assignment_and_holds_short_circuit_success()
    -> Result<()> {
        let mut graph = pipelined_test_graph().await;
        let executor = mock_executor("producer".into());
        let first = graph.pop_next_task(&executor.id)?.unwrap();
        let producer = first.key.stage_id;
        assert!(graph.fetch_running_stage_for_admission(&[], true).is_none());
        graph.update_task_status(
            &executor,
            vec![mock_completed_task(first, &executor.id)],
            4,
            4,
        )?;
        // Committed input alone cannot bypass unscheduled producer work.
        assert!(graph.fetch_running_stage_for_admission(&[], true).is_none());
        let straggler = graph.pop_next_task(&executor.id)?.unwrap();
        assert_eq!(straggler.key.stage_id, producer);
        let consumer_executor = mock_executor("consumer".into());
        let consumer = bind_tail(&mut graph, &consumer_executor.id);
        let consumer_id = consumer.key.stage_id;
        assert!(!graph.stages[&consumer_id].output_links().is_empty());
        graph.update_task_status(
            &consumer_executor,
            vec![mock_completed_task(consumer, &consumer_executor.id)],
            4,
            4,
        )?;
        assert_eq!(graph.deferred_successes.len(), 1);
        assert!(
            !graph
                .shuffle_inputs
                .lock()
                .get(&graph.job_id, consumer_id)
                .unwrap()
                .has_committed_input()
        );
        graph.update_task_status(
            &executor,
            vec![mock_completed_task(straggler, &executor.id)],
            4,
            4,
        )?;
        assert!(graph.deferred_successes.is_empty());
        assert!(
            graph
                .shuffle_inputs
                .lock()
                .get(&graph.job_id, consumer_id)
                .unwrap()
                .has_committed_input()
        );
        Ok(())
    }

    #[tokio::test]
    async fn pipelined_unpublished_retry_revokes_consumers_without_generation_change()
    -> Result<()> {
        let mut graph = pipelined_test_graph().await;
        let executor = mock_executor("producer".into());
        let first = graph.pop_next_task(&executor.id)?.unwrap();
        let producer = first.key.stage_id;
        graph.update_task_status(
            &executor,
            vec![mock_completed_task(first, &executor.id)],
            4,
            4,
        )?;
        let straggler = graph.pop_next_task(&executor.id)?.unwrap();
        let consumer = bind_tail(&mut graph, "consumer");
        let events = graph.update_task_status(
            &executor,
            vec![mock_failed_task(
                straggler,
                FailedTask {
                    error: "injected transient failure".into(),
                    retryable: true,
                    count_to_failures: false,
                    failed_reason: Some(failed_task::FailedReason::IoError(IoError {})),
                },
            )],
            4,
            4,
        )?;
        assert!(events.iter().any(|event| matches!(event, QueryStageSchedulerEvent::CancelTasks(tasks) if tasks.iter().any(|t| t.stage_id == consumer.key.stage_id))));
        assert_eq!(
            graph
                .shuffle_inputs
                .lock()
                .get(&graph.job_id, producer)
                .unwrap()
                .generation(),
            1
        );
        assert!(graph.fetch_running_stage_for_admission(&[], true).is_none());
        assert_eq!(
            graph
                .fetch_running_stage_for_admission(&[], false)
                .unwrap()
                .stage_id,
            producer
        );
        Ok(())
    }

    #[tokio::test]
    async fn pipelined_fetch_failure_retires_output_from_running_producer() -> Result<()>
    {
        let mut graph = pipelined_test_graph().await;
        let producer_executor = mock_executor("producer-with-lost-output".into());

        let first = graph.pop_next_task(&producer_executor.id)?.unwrap();
        let producer_stage = first.key.stage_id;
        let first_task_id = first.key.task_id;
        graph.update_task_status(
            &producer_executor,
            vec![mock_completed_task(first, &producer_executor.id)],
            4,
            4,
        )?;

        // Assign the producer tail without completing it. The producer stage
        // therefore remains Running while downstream consumes the first task's
        // committed output.
        let _straggler = graph.pop_next_task("producer-straggler")?.unwrap();
        let consumer_executor = mock_executor("consumer".into());
        let consumer = bind_tail(&mut graph, &consumer_executor.id);
        let consumer_stage = consumer.key.stage_id;

        let (lost_executor, lost_output_partition) = {
            let output = graph
                .shuffle_inputs
                .lock()
                .stage_output(&graph.job_id, producer_stage)
                .expect("published producer output");
            let location = output
                .partition_locations
                .values()
                .flat_map(|locations| locations.iter())
                .find(|location| location.map_partition_id == first_task_id)
                .expect("first producer task must have a committed block");
            (
                location.executor_meta.id.clone(),
                location.partition_id.partition_id,
            )
        };

        let events = graph.update_task_status(
            &consumer_executor,
            vec![mock_failed_task(
                consumer,
                wrapped_fetch_failed_task(
                    &lost_executor,
                    producer_stage,
                    lost_output_partition,
                ),
            )],
            4,
            4,
        )?;

        // The task reporting FetchPartitionError is already terminal, so there
        // is nothing to send back to its executor as a cancellation. What must
        // happen is that the whole generation-pinned consumer stage is revoked;
        // any still-running sibling tasks would be returned in CancelTasks.
        assert!(
            !events.iter().any(|event| matches!(
                event,
                QueryStageSchedulerEvent::JobRunningFailed { .. }
            )),
            "materialized result loss is recoverable"
        );
        assert!(
            matches!(
                graph.stages.get(&consumer_stage),
                Some(ExecutionStage::UnResolved(_))
            ),
            "the generation-pinned consumer attempt must be rolled back"
        );

        let ExecutionStage::Running(producer) = &graph.stages[&producer_stage] else {
            panic!("producer must remain Running and retry the lost task");
        };
        assert!(matches!(
            producer.task_infos[first_task_id].task_status,
            task_status::Status::Failed(FailedTask {
                failed_reason: Some(failed_task::FailedReason::ResultLost(_)),
                ..
            })
        ));
        assert!(
            producer.pending.remaining()
                >= producer.task_infos[first_task_id]
                    .global_input_partition_ids
                    .len(),
            "the lost producer slice must be returned to pending"
        );
        assert_eq!(
            graph
                .shuffle_inputs
                .lock()
                .get(&graph.job_id, producer_stage)
                .unwrap()
                .generation(),
            2,
            "dropping an accepted publication must roll the generation"
        );

        Ok(())
    }

    #[tokio::test]
    async fn pipelined_visible_output_loss_invalidates_and_releases_capacity()
    -> Result<()> {
        let mut graph = pipelined_test_graph().await;
        let executor = mock_executor("lost".into());
        let first = graph.pop_next_task(&executor.id)?.unwrap();
        let producer = first.key.stage_id;
        graph.update_task_status(
            &executor,
            vec![mock_completed_task(first, &executor.id)],
            4,
            4,
        )?;
        let _straggler = graph.pop_next_task("survivor")?.unwrap();
        let consumer = bind_tail(&mut graph, "consumer");
        let (_, cancelled) = graph.reset_stages_on_lost_executor(&executor.id)?;
        assert!(
            cancelled
                .iter()
                .any(|task| task.stage_id == consumer.key.stage_id)
        );
        assert_eq!(
            graph
                .shuffle_inputs
                .lock()
                .get(&graph.job_id, producer)
                .unwrap()
                .generation(),
            2
        );
        assert_eq!(
            graph
                .fetch_running_stage_for_admission(&[], false)
                .unwrap()
                .stage_id,
            producer
        );
        Ok(())
    }

    #[derive(Debug)]
    struct FailingPlanRewriteExec {
        input: Arc<dyn ExecutionPlan>,
    }

    #[derive(Debug)]
    struct EligibilityChildrenExec {
        inputs: Vec<Arc<dyn ExecutionPlan>>,
        properties: Arc<PlanProperties>,
    }

    impl EligibilityChildrenExec {
        fn new(inputs: Vec<Arc<dyn ExecutionPlan>>) -> Self {
            assert!(!inputs.is_empty());
            let properties = inputs[0].properties().clone();
            Self { inputs, properties }
        }
    }

    impl DisplayAs for EligibilityChildrenExec {
        fn fmt_as(
            &self,
            _t: DisplayFormatType,
            f: &mut std::fmt::Formatter,
        ) -> std::fmt::Result {
            write!(f, "EligibilityChildrenExec")
        }
    }

    impl ExecutionPlan for EligibilityChildrenExec {
        fn name(&self) -> &str {
            "EligibilityChildrenExec"
        }

        fn properties(&self) -> &Arc<PlanProperties> {
            &self.properties
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            self.inputs.iter().collect()
        }

        fn apply_expressions(
            &self,
            _f: &mut dyn FnMut(
                &Arc<dyn PhysicalExpr>,
            ) -> DataFusionResult<TreeNodeRecursion>,
        ) -> DataFusionResult<TreeNodeRecursion> {
            Ok(TreeNodeRecursion::Continue)
        }

        fn with_new_children(
            self: Arc<Self>,
            children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
            Ok(Arc::new(Self::new(children)))
        }

        fn execute(
            &self,
            _partition: usize,
            _context: Arc<TaskContext>,
        ) -> DataFusionResult<SendableRecordBatchStream> {
            Err(DataFusionError::Plan(
                "EligibilityChildrenExec is test-only".to_string(),
            ))
        }
    }

    impl DisplayAs for FailingPlanRewriteExec {
        fn fmt_as(
            &self,
            _t: DisplayFormatType,
            f: &mut std::fmt::Formatter,
        ) -> std::fmt::Result {
            write!(f, "FailingPlanRewriteExec")
        }
    }

    impl ExecutionPlan for FailingPlanRewriteExec {
        fn name(&self) -> &str {
            "FailingPlanRewriteExec"
        }

        fn properties(&self) -> &Arc<PlanProperties> {
            self.input.properties()
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            vec![&self.input]
        }

        fn apply_expressions(
            &self,
            _f: &mut dyn FnMut(
                &Arc<dyn PhysicalExpr>,
            ) -> DataFusionResult<TreeNodeRecursion>,
        ) -> DataFusionResult<TreeNodeRecursion> {
            Ok(TreeNodeRecursion::Continue)
        }

        fn with_new_children(
            self: Arc<Self>,
            _children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
            Err(DataFusionError::Internal(
                "forced plan rewrite failure".to_owned(),
            ))
        }

        fn execute(
            &self,
            partition: usize,
            context: Arc<TaskContext>,
        ) -> DataFusionResult<SendableRecordBatchStream> {
            self.input.execute(partition, context)
        }
    }

    fn fail_plan_rewrites(plan: &mut Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        let failing_plan: Arc<dyn ExecutionPlan> = Arc::new(FailingPlanRewriteExec {
            input: Arc::clone(plan),
        });
        *plan = Arc::clone(&failing_plan);
        failing_plan
    }

    #[tokio::test]
    async fn test_intermediate_stage_ids() {
        // A simple aggregation produces a 2-stage graph: one intermediate
        // stage (non-empty output_links) feeding one final stage (empty
        // output_links).
        let graph = test_aggregation_plan(4).await;

        assert_eq!(graph.stages().len(), 2);

        // Exactly one final stage.
        let final_count = graph
            .stages()
            .values()
            .filter(|s| s.output_links().is_empty())
            .count();
        assert_eq!(final_count, 1);

        // Intermediate = all - final = exactly one stage, and none of the
        // returned ids is a final stage.
        let intermediate = graph.intermediate_stage_ids();
        assert_eq!(intermediate.len(), 1);
        for id in &intermediate {
            let stage = graph.stages().get(&(*id as usize)).unwrap();
            assert!(!stage.output_links().is_empty());
        }
    }

    #[tokio::test]
    async fn test_resolve_stage_preserves_stage_on_plan_rewrite_error() -> Result<()> {
        let mut graph = test_aggregation_plan(4).await;
        let stage_id = graph
            .stages
            .iter()
            .find_map(|(stage_id, stage)| {
                matches!(stage, ExecutionStage::UnResolved(_)).then_some(*stage_id)
            })
            .expect("expected an unresolved stage");
        let stage_count = graph.stage_count();
        let original_plan = match graph.stages.get_mut(&stage_id) {
            Some(ExecutionStage::UnResolved(stage)) => {
                for input in stage.inputs.values_mut() {
                    input.complete = true;
                }
                assert!(stage.resolvable());
                fail_plan_rewrites(&mut stage.plan)
            }
            _ => unreachable!(),
        };

        assert!(graph.resolve_stage(stage_id).is_err());
        assert_eq!(graph.stage_count(), stage_count);
        match graph.stages.get(&stage_id) {
            Some(ExecutionStage::UnResolved(stage)) => {
                assert!(Arc::ptr_eq(&stage.plan, &original_plan));
            }
            _ => panic!("expected the original unresolved stage"),
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_rollback_resolved_stage_preserves_stage_on_plan_rewrite_error()
    -> Result<()> {
        let mut graph = test_aggregation_plan(4).await;
        revive_graph_and_complete_next_stage(&mut graph)?;
        let stage_id = graph
            .stages
            .iter()
            .find_map(|(stage_id, stage)| {
                matches!(stage, ExecutionStage::Resolved(stage) if !stage.inputs.is_empty())
                    .then_some(*stage_id)
            })
            .expect("expected a resolved stage with inputs");
        let stage_count = graph.stage_count();
        let (original_plan, original_attempt) = match graph.stages.get_mut(&stage_id) {
            Some(ExecutionStage::Resolved(stage)) => {
                (fail_plan_rewrites(&mut stage.plan), stage.stage_attempt_num)
            }
            _ => unreachable!(),
        };

        assert!(graph.rollback_resolved_stage(stage_id).is_err());
        assert_eq!(graph.stage_count(), stage_count);
        match graph.stages.get(&stage_id) {
            Some(ExecutionStage::Resolved(stage)) => {
                assert_eq!(stage.stage_attempt_num, original_attempt);
                assert!(Arc::ptr_eq(&stage.plan, &original_plan));
            }
            _ => panic!("expected the original resolved stage"),
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_rollback_running_stage_preserves_stage_on_plan_rewrite_error()
    -> Result<()> {
        let mut graph = test_aggregation_plan(4).await;
        revive_graph_and_complete_next_stage(&mut graph)?;
        let stage_id = graph
            .stages
            .iter()
            .find_map(|(stage_id, stage)| {
                matches!(stage, ExecutionStage::Resolved(stage) if !stage.inputs.is_empty())
                    .then_some(*stage_id)
            })
            .expect("expected a resolved stage with inputs");
        assert!(graph.revive());
        assert!(graph.pop_next_task("executor-id")?.is_some());
        let stage_count = graph.stage_count();
        let (original_plan, original_attempt, original_pending, original_tasks) =
            match graph.stages.get_mut(&stage_id) {
                Some(ExecutionStage::Running(stage)) => (
                    fail_plan_rewrites(&mut stage.plan),
                    stage.stage_attempt_num,
                    stage.available_tasks(),
                    stage.running_tasks(),
                ),
                _ => unreachable!(),
            };

        assert!(
            graph
                .rollback_running_stage(
                    stage_id,
                    HashSet::from(["executor-id".to_owned()]),
                )
                .is_err()
        );
        assert_eq!(graph.stage_count(), stage_count);
        match graph.stages.get(&stage_id) {
            Some(ExecutionStage::Running(stage)) => {
                assert_eq!(stage.stage_attempt_num, original_attempt);
                assert_eq!(stage.available_tasks(), original_pending);
                assert_eq!(stage.running_tasks(), original_tasks);
                assert!(Arc::ptr_eq(&stage.plan, &original_plan));
            }
            _ => panic!("expected the original running stage"),
        }

        Ok(())
    }

    #[test]
    fn test_pipelined_shuffle_shape_eligibility_matrix() {
        let cases = [
            ("static", false, false, false, false, true),
            ("broadcast", true, false, false, false, false),
            ("coalesced", false, true, false, false, false),
            ("range-reader", false, false, true, false, false),
            ("range-writer", false, false, false, true, false),
        ];

        for (name, broadcast, coalesced, range_reader, range_writer, expected) in cases {
            assert_eq!(
                super::pipelined_shape_eligible(
                    broadcast,
                    coalesced,
                    range_reader,
                    range_writer,
                ),
                expected,
                "unexpected eligibility for {name}"
            );
        }
    }

    #[test]
    fn test_pipelined_plan_rejects_mixed_multi_input_when_one_edge_is_ineligible() {
        let schema = Arc::new(datafusion::arrow::datatypes::Schema::empty());
        let eligible: Arc<dyn ExecutionPlan> = Arc::new(UnresolvedShuffleExec::new(
            1,
            schema.clone(),
            Partitioning::UnknownPartitioning(2),
        ));
        let broadcast: Arc<dyn ExecutionPlan> =
            Arc::new(UnresolvedShuffleExec::new_broadcast(2, schema, 2));

        let all_static =
            EligibilityChildrenExec::new(vec![eligible.clone(), eligible.clone()]);
        assert!(super::pipelined_plan_eligible(&all_static));

        let mixed = EligibilityChildrenExec::new(vec![eligible, broadcast]);
        assert!(!super::pipelined_plan_eligible(&mixed));
    }

    #[tokio::test]
    async fn test_final_stage_remains_behind_completion_barrier() -> Result<()> {
        let config = Arc::new(
            SessionConfig::new_with_ballista()
                .with_ballista_shuffle_pipelined_enabled(true)
                .with_ballista_adaptive_query_planner(false),
        );
        let job = JobId::from("pipelined-final-barrier");
        let graph = test_aggregation_plan_with_config(2, &job, config).await;

        let final_stage = graph
            .stages
            .values()
            .find_map(|stage| match stage {
                ExecutionStage::UnResolved(stage)
                    if stage.output_links.is_empty() && !stage.inputs.is_empty() =>
                {
                    Some(stage)
                }
                _ => None,
            })
            .expect("expected an unresolved final stage");

        assert!(
            graph.pipelined_handles(final_stage).is_none(),
            "final stage must never be admitted before producer completion"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_failed_shuffle_publication_does_not_accept_task_success() -> Result<()>
    {
        let config = Arc::new(
            SessionConfig::new_with_ballista()
                .with_ballista_shuffle_pipelined_enabled(true)
                .with_ballista_adaptive_query_planner(false),
        );
        let job_id = JobId::from("pipelined-rejected-publication");
        let executor = mock_executor("executor-id1".to_string());
        let mut graph = test_aggregation_plan_with_config(2, &job_id, config).await;
        graph.revive();

        let task = graph
            .pop_next_task(&executor.id)?
            .expect("expected a running producer task");
        let stage_id = task.key.stage_id;
        let task_id = task.key.task_id;

        {
            let mut registry = graph.shuffle_inputs.lock();
            let generation = registry
                .get(&job_id, stage_id)
                .expect("registry state")
                .generation();
            registry.seal(&job_id, stage_id, generation)?;
        }

        let task_status = mock_completed_task(task, &executor.id);
        assert!(
            graph
                .update_task_status(&executor, vec![task_status], 1, 1)
                .is_err(),
            "publication into a sealed generation must fail"
        );

        let ExecutionStage::Running(stage) = graph
            .stages
            .get(&stage_id)
            .expect("stage should remain running")
        else {
            panic!("failed publication must not transition the stage");
        };
        assert!(matches!(
            stage.task_infos[task_id].task_status,
            task_status::Status::Running(_)
        ));
        assert!(
            stage.stage_metrics.is_none(),
            "metrics must not be applied before publication is accepted"
        );
        assert!(stage.runtime_stats_reports.is_empty());
        assert!(stage.window_state_reports.is_empty());

        let registry = graph.shuffle_inputs.lock();
        let state = registry
            .get(&job_id, stage_id)
            .expect("registry state should remain present");
        assert!(!state.has_committed_input());

        Ok(())
    }

    #[tokio::test]
    async fn test_pipelined_success_replay_does_not_publish_twice() -> Result<()> {
        use crate::state::shuffle_input::ShuffleInputRead;

        let config = Arc::new(
            SessionConfig::new_with_ballista()
                .with_ballista_shuffle_pipelined_enabled(true)
                .with_ballista_adaptive_query_planner(false),
        );
        let job = JobId::from("pipelined-replay");
        let mut graph = test_aggregation_plan_with_config(2, &job, config).await;
        graph.revive();

        let executor = mock_executor("executor-id1".to_string());
        let task = graph
            .pop_next_task(&executor.id)?
            .expect("expected a running producer task");
        let stage_id = task.key.stage_id;
        let status = mock_completed_task(task, &executor.id);

        graph.update_task_status(&executor, vec![status.clone()], 1, 1)?;
        let (generation, first_version) = {
            let registry = graph.shuffle_inputs.lock();
            let state = registry.get(&job, stage_id).expect("registry state");
            let generation = state.generation();
            let ShuffleInputRead::Update(snapshot) =
                registry.read(&job, stage_id, generation, 0, &[])?
            else {
                panic!("expected metadata update");
            };
            (generation, snapshot.version)
        };

        graph.update_task_status(&executor, vec![status], 1, 1)?;
        let second_version = {
            let registry = graph.shuffle_inputs.lock();
            let ShuffleInputRead::Update(snapshot) =
                registry.read(&job, stage_id, generation, 0, &[])?
            else {
                panic!("expected metadata update");
            };
            snapshot.version
        };

        assert_eq!(first_version, second_version);
        Ok(())
    }

    #[tokio::test]
    async fn test_mismatched_stage_attempt_success_cannot_publish() -> Result<()> {
        let config = Arc::new(
            SessionConfig::new_with_ballista()
                .with_ballista_shuffle_pipelined_enabled(true)
                .with_ballista_adaptive_query_planner(false),
        );
        let job = JobId::from("pipelined-stale-attempt");
        let mut graph = test_aggregation_plan_with_config(2, &job, config).await;
        graph.revive();

        let executor = mock_executor("executor-id1".to_string());
        let task = graph
            .pop_next_task(&executor.id)?
            .expect("expected a running producer task");
        let stage_id = task.key.stage_id;
        let task_id = task.key.task_id;
        let mut status = mock_completed_task(task, &executor.id);

        let ExecutionStage::Running(stage) =
            graph.stages.get_mut(&stage_id).expect("running stage")
        else {
            panic!("expected running stage");
        };
        stage.stage_attempt_num += 1;
        status.stage_attempt_num = (stage.stage_attempt_num - 1) as u32;

        graph.update_task_status(&executor, vec![status.clone()], 1, 1)?;

        // Active pipelining also rejects a future/mismatched attempt rather
        // than accepting metadata that cannot belong to the current stage.
        status.stage_attempt_num = (status.stage_attempt_num + 2) as u32;
        graph.update_task_status(&executor, vec![status], 1, 1)?;

        let ExecutionStage::Running(stage) =
            graph.stages.get(&stage_id).expect("running stage")
        else {
            panic!("stale status must not transition the stage");
        };
        assert!(matches!(
            stage.task_infos[task_id].task_status,
            task_status::Status::Running(_)
        ));
        let registry = graph.shuffle_inputs.lock();
        assert!(
            !registry
                .get(&job, stage_id)
                .expect("registry state")
                .has_committed_input()
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_valid_inflight_task_publishes_into_current_generation_after_rollover()
    -> Result<()> {
        use crate::state::shuffle_input::ShuffleInputRead;

        let config = Arc::new(
            SessionConfig::new_with_ballista()
                .with_ballista_shuffle_pipelined_enabled(true)
                .with_ballista_adaptive_query_planner(false),
        );
        let job = JobId::from("pipelined-rollover-survivor");
        let mut graph = test_aggregation_plan_with_config(2, &job, config).await;
        graph.revive();

        let executor = mock_executor("executor-id1".to_string());
        let task = graph
            .pop_next_task(&executor.id)?
            .expect("expected a running producer task");
        let stage_id = task.key.stage_id;

        let generation = {
            let mut registry = graph.shuffle_inputs.lock();
            let old = registry
                .get(&job, stage_id)
                .expect("registry state")
                .generation();
            registry.invalidate(&job, stage_id, old, &HashSet::new())?
        };

        let status = mock_completed_task(task, &executor.id);
        graph.update_task_status(&executor, vec![status], 1, 1)?;

        let registry = graph.shuffle_inputs.lock();
        let ShuffleInputRead::Update(snapshot) =
            registry.read(&job, stage_id, generation, 0, &[0, 1])?
        else {
            panic!("surviving task must publish into the current generation");
        };
        assert_eq!(snapshot.generation, generation);
        assert!(!snapshot.locations.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_active_pipelining_fails_closed_if_registry_state_is_missing()
    -> Result<()> {
        let config = Arc::new(
            SessionConfig::new_with_ballista()
                .with_ballista_shuffle_pipelined_enabled(true)
                .with_ballista_adaptive_query_planner(false),
        );
        let job = JobId::from("pipelined-missing-registry");
        let mut graph = test_aggregation_plan_with_config(2, &job, config).await;
        graph.revive();

        let executor = mock_executor("executor-id1".to_string());
        let task = graph
            .pop_next_task(&executor.id)?
            .expect("expected a running producer task");
        let stage_id = task.key.stage_id;
        let task_id = task.key.task_id;

        graph.shuffle_inputs.lock().close_job(&job);

        let task_status = mock_completed_task(task, &executor.id);
        let error = graph
            .update_task_status(&executor, vec![task_status], 1, 1)
            .expect_err("missing active registry state must fail closed");
        assert!(
            error
                .to_string()
                .contains("pipelined shuffle input missing")
        );

        let ExecutionStage::Running(stage) = graph
            .stages
            .get(&stage_id)
            .expect("stage should remain running")
        else {
            panic!("missing publication state must not transition the stage");
        };
        assert!(matches!(
            stage.task_infos[task_id].task_status,
            task_status::Status::Running(_)
        ));

        Ok(())
    }

    #[tokio::test]
    async fn test_pipelining_disabled_does_not_initialize_shuffle_registry() -> Result<()>
    {
        let graph = test_aggregation_plan(2).await;
        let registry = graph.shuffle_inputs.lock();
        for stage_id in graph.stages.keys() {
            assert!(
                registry.get(&graph.job_id, *stage_id).is_none(),
                "feature-off graph must retain the legacy scheduler path"
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_fail_job_sets_end_time_and_failed_metadata() -> Result<()> {
        let mut graph = test_aggregation_plan(4).await;
        let start = graph.start_time();
        assert_eq!(graph.end_time(), 0);

        ExecutionGraph::fail_job(&mut graph, "test failure".to_string());

        assert!(
            matches!(
                graph.status().status.as_ref(),
                Some(job_status::Status::Failed(f)) if f.error == "test failure"
            ),
            "expected FailedJob status after fail_job"
        );
        assert!(
            graph.end_time() >= start,
            "end_time ({}) should be set and >= start_time ({})",
            graph.end_time(),
            start
        );

        if let Some(job_status::Status::Failed(failed)) = &graph.status().status {
            assert_eq!(failed.started_at, start);
            assert_eq!(failed.ended_at, graph.end_time());
        } else {
            panic!("missing FailedJob");
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_drain_tasks() -> Result<()> {
        let mut agg_graph = test_aggregation_plan(4).await;

        println!("Graph: {agg_graph:?}");

        drain_tasks(&mut agg_graph)?;

        assert!(
            agg_graph.is_successful(),
            "Failed to complete aggregation plan"
        );

        let mut coalesce_graph = test_coalesce_plan(4).await;

        drain_tasks(&mut coalesce_graph)?;

        assert!(
            coalesce_graph.is_successful(),
            "Failed to complete coalesce plan"
        );

        let mut join_graph = test_join_plan(4).await;

        drain_tasks(&mut join_graph)?;

        println!("{join_graph:?}");

        assert!(join_graph.is_successful(), "Failed to complete join plan");

        let mut union_all_graph = test_union_all_plan(4).await;

        drain_tasks(&mut union_all_graph)?;

        println!("{union_all_graph:?}");

        assert!(
            union_all_graph.is_successful(),
            "Failed to complete union plan"
        );

        let mut union_graph = test_union_plan(4).await;

        drain_tasks(&mut union_graph)?;

        println!("{union_graph:?}");

        assert!(union_graph.is_successful(), "Failed to complete union plan");

        Ok(())
    }

    #[tokio::test]
    async fn test_finalize() -> Result<()> {
        let mut agg_graph = test_aggregation_plan(4).await;

        drain_tasks(&mut agg_graph)?;

        let status = agg_graph.status();

        assert!(matches!(
            status,
            protobuf::JobStatus {
                status: Some(job_status::Status::Successful(_)),
                ..
            }
        ));

        let outputs = agg_graph.output_locations();

        for location in outputs {
            assert_eq!(location.executor_meta.host, "localhost2".to_owned());
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_reset_completed_stage_executor_lost() -> Result<()> {
        let executor1 = mock_executor("executor-id1".to_string());
        let executor2 = mock_executor("executor-id2".to_string());
        let mut join_graph = test_join_plan(4).await;

        // With the improvement of https://github.com/apache/arrow-datafusion/pull/4122,
        // unnecessary RepartitionExec can be removed
        assert_eq!(join_graph.stage_count(), 4);
        assert_eq!(join_graph.available_tasks(), 0);

        // Call revive to move the two leaf Resolved stages to Running
        join_graph.revive();

        assert_eq!(join_graph.stage_count(), 4);
        assert_eq!(join_graph.available_tasks(), 4);

        // Complete the first stage
        revive_graph_and_complete_next_stage_with_executor(&mut join_graph, &executor1)?;

        // Complete the second stage
        revive_graph_and_complete_next_stage_with_executor(&mut join_graph, &executor2)?;

        join_graph.revive();
        // There are 4 tasks pending schedule for the 3rd stage
        assert_eq!(join_graph.available_tasks(), 4);

        // Complete 1 task
        if let Some(task) = join_graph.pop_next_task(&executor1.id)? {
            let task_status = mock_completed_task(task, &executor1.id);
            join_graph.update_task_status(&executor1, vec![task_status], 1, 1)?;
        }
        // Mock 1 running task
        let _task = join_graph.pop_next_task(&executor1.id)?;

        let reset = join_graph.reset_stages_on_lost_executor(&executor1.id)?;

        // Two stages were reset, 1 Running stage rollback to Unresolved and 1 Completed stage move to Running
        assert_eq!(reset.0.len(), 2);
        assert_eq!(join_graph.available_tasks(), 2);

        drain_tasks(&mut join_graph)?;
        assert!(join_graph.is_successful(), "Failed to complete join plan");

        Ok(())
    }

    #[tokio::test]
    async fn test_reset_resolved_stage_executor_lost() -> Result<()> {
        let executor1 = mock_executor("executor-id1".to_string());
        let executor2 = mock_executor("executor-id2".to_string());
        let mut join_graph = test_join_plan(4).await;

        assert_eq!(join_graph.stage_count(), 4);
        assert_eq!(join_graph.available_tasks(), 0);

        // Call revive to move the two leaf Resolved stages to Running
        join_graph.revive();

        assert_eq!(join_graph.stage_count(), 4);
        assert_eq!(join_graph.available_tasks(), 4);

        // Complete the first stage
        assert_eq!(revive_graph_and_complete_next_stage(&mut join_graph)?, 2);

        // Complete the second stage
        assert_eq!(
            revive_graph_and_complete_next_stage_with_executor(
                &mut join_graph,
                &executor2
            )?,
            2
        );

        // There are 0 tasks pending schedule now
        assert_eq!(join_graph.available_tasks(), 0);

        let reset = join_graph.reset_stages_on_lost_executor(&executor1.id)?;

        // Two stages were reset, 1 Resolved stage rollback to Unresolved and 1 Completed stage move to Running
        assert_eq!(reset.0.len(), 2);
        assert_eq!(join_graph.available_tasks(), 2);

        drain_tasks(&mut join_graph)?;
        assert!(join_graph.is_successful(), "Failed to complete join plan");

        Ok(())
    }

    #[tokio::test]
    async fn test_task_update_after_reset_stage() -> Result<()> {
        let executor1 = mock_executor("executor-id1".to_string());
        let executor2 = mock_executor("executor-id2".to_string());
        let mut agg_graph = test_aggregation_plan(4).await;

        assert_eq!(agg_graph.stage_count(), 2);
        assert_eq!(agg_graph.available_tasks(), 0);

        // Call revive to move the leaf Resolved stages to Running
        agg_graph.revive();

        assert_eq!(agg_graph.stage_count(), 2);
        assert_eq!(agg_graph.available_tasks(), 2);

        // Complete the first stage
        revive_graph_and_complete_next_stage_with_executor(&mut agg_graph, &executor1)?;

        // 1st task in the second stage
        if let Some(task) = agg_graph.pop_next_task(&executor2.id)? {
            let task_status = mock_completed_task(task, &executor2.id);
            agg_graph.update_task_status(&executor2, vec![task_status], 1, 1)?;
        }

        // 2rd task in the second stage
        if let Some(task) = agg_graph.pop_next_task(&executor1.id)? {
            let task_status = mock_completed_task(task, &executor1.id);
            agg_graph.update_task_status(&executor1, vec![task_status], 1, 1)?;
        }

        // 3rd task in the second stage, scheduled but not completed
        let task = agg_graph.pop_next_task(&executor1.id)?;

        // There is 1 task pending schedule now
        assert_eq!(agg_graph.available_tasks(), 1);

        let reset = agg_graph.reset_stages_on_lost_executor(&executor1.id)?;

        // 3rd task status update comes later.
        let task_status = mock_completed_task(task.unwrap(), &executor1.id);
        agg_graph.update_task_status(&executor1, vec![task_status], 1, 1)?;

        // Two stages were reset, 1 Running stage rollback to Unresolved and 1 Completed stage move to Running
        assert_eq!(reset.0.len(), 2);
        assert_eq!(agg_graph.available_tasks(), 2);

        // Call the reset again
        let reset = agg_graph.reset_stages_on_lost_executor(&executor1.id)?;
        assert_eq!(reset.0.len(), 0);
        assert_eq!(agg_graph.available_tasks(), 2);

        drain_tasks(&mut agg_graph)?;
        assert!(agg_graph.is_successful(), "Failed to complete agg plan");

        Ok(())
    }

    #[tokio::test]
    async fn test_do_not_retry_killed_task() -> Result<()> {
        let executor = mock_executor("executor-id-123".to_string());
        let mut agg_graph = test_aggregation_plan(4).await;
        // Call revive to move the leaf Resolved stages to Running
        agg_graph.revive();

        // Complete the first stage
        revive_graph_and_complete_next_stage(&mut agg_graph)?;

        // 1st task in the second stage
        let task1 = agg_graph.pop_next_task(&executor.id)?.unwrap();
        let task_status1 = mock_completed_task(task1, &executor.id);

        // 2rd task in the second stage
        let task2 = agg_graph.pop_next_task(&executor.id)?.unwrap();
        let task_status2 = mock_failed_task(
            task2,
            FailedTask {
                error: "Killed".to_string(),
                retryable: false,
                count_to_failures: false,
                failed_reason: Some(failed_task::FailedReason::TaskKilled(TaskKilled {})),
            },
        );

        agg_graph.update_task_status(
            &executor,
            vec![task_status1, task_status2],
            4,
            4,
        )?;

        assert_eq!(agg_graph.available_tasks(), 2);
        drain_tasks(&mut agg_graph)?;
        assert_eq!(agg_graph.available_tasks(), 0);

        assert!(
            !agg_graph.is_successful(),
            "Expected the agg graph can not complete"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_max_task_failed_count() -> Result<()> {
        let executor = mock_executor("executor-id2".to_string());
        let mut agg_graph = test_aggregation_plan(2).await;
        // Call revive to move the leaf Resolved stages to Running
        agg_graph.revive();

        // Complete the first stage
        revive_graph_and_complete_next_stage(&mut agg_graph)?;

        // 1st task in the second stage
        let task1 = agg_graph.pop_next_task(&executor.id)?.unwrap();
        let task_status1 = mock_completed_task(task1, &executor.id);

        // 2rd task in the second stage, failed due to IOError
        let task2 = agg_graph.pop_next_task(&executor.id)?.unwrap();
        let task_status2 = mock_failed_task(
            task2.clone(),
            FailedTask {
                error: "IOError".to_string(),
                retryable: true,
                count_to_failures: true,
                failed_reason: Some(failed_task::FailedReason::IoError(IoError {})),
            },
        );

        agg_graph.update_task_status(
            &executor,
            vec![task_status1, task_status2],
            4,
            4,
        )?;

        assert_eq!(agg_graph.available_tasks(), 1);

        let mut last_attempt = 0;
        // 2rd task's attempts.
        //
        // Under the append-only task_infos model, each retry gets a fresh
        // task_id (rather than reusing the original task's slot). The
        // global_input_partition_ids is what stably identifies "which task is being
        // retried" — assert on that instead of task_id.
        for attempt in 1..5 {
            if let Some(task2_attempt) = agg_graph.pop_next_task(&executor.id)? {
                assert_eq!(
                    task2_attempt.global_input_partition_ids,
                    task2.global_input_partition_ids
                );
                assert_eq!(task2_attempt.task_attempt, attempt);
                last_attempt = task2_attempt.task_attempt;
                let task_status = mock_failed_task(
                    task2_attempt.clone(),
                    FailedTask {
                        error: "IOError".to_string(),
                        retryable: true,
                        count_to_failures: true,
                        failed_reason: Some(failed_task::FailedReason::IoError(
                            IoError {},
                        )),
                    },
                );
                agg_graph.update_task_status(&executor, vec![task_status], 4, 4)?;
            }
        }

        assert!(
            matches!(
                agg_graph.status(),
                JobStatus {
                    status: Some(job_status::Status::Failed(_)),
                    ..
                }
            ),
            "Expected job status to be Failed"
        );

        assert_eq!(last_attempt, 3);

        let failure_reason = format!("{:?}", agg_graph.status);
        assert!(failure_reason.contains(
            "Task 1 in Stage 2 failed 4 times, fail the stage, most recent failure reason"
        ));
        assert!(failure_reason.contains("IOError"));
        assert!(!agg_graph.is_successful());

        Ok(())
    }

    // Aborting a running job (failure or cancellation) must transition every
    // running stage to Failed and return its in-flight tasks for cancellation.
    // `abort_running` is the shared teardown invoked by `abort_job`.
    #[tokio::test]
    async fn test_abort_running_cancels_stages_and_returns_inflight_tasks() -> Result<()>
    {
        let executor = mock_executor("executor-id1".to_string());
        let mut graph = test_join_plan(2).await;

        // Call revive to move the two leaf Resolved stages to Running
        graph.revive();
        assert!(
            graph.running_stages().len() >= 2,
            "expected two concurrently running leaf stages, found {:?}",
            graph.running_stages()
        );

        // Dispatch a task so there is an in-flight task to cancel
        let _task = graph.pop_next_task(&executor.id)?.unwrap();

        // Aborting cancels every running stage and returns its in-flight tasks
        let cancelled = graph.abort_running("job aborted".to_string());

        assert!(
            !cancelled.is_empty(),
            "abort_running must return the in-flight tasks to cancel"
        );
        assert!(
            graph.running_stages().is_empty(),
            "every running stage must be cancelled, found {:?}",
            graph.running_stages()
        );
        assert!(
            matches!(
                graph.status(),
                JobStatus {
                    status: Some(job_status::Status::Failed(_)),
                    ..
                }
            ),
            "the job must be Failed after abort"
        );

        // In-flight tasks of the cancelled stage are recorded as Failed(TaskKilled)
        let has_killed_task = graph.stages.values().any(|stage| match stage {
            ExecutionStage::Failed(failed) => failed.task_infos.iter().any(|info| {
                matches!(
                    &info.task_status,
                    task_status::Status::Failed(FailedTask {
                        failed_reason: Some(failed_task::FailedReason::TaskKilled(_)),
                        ..
                    })
                )
            }),
            _ => false,
        });
        assert!(
            has_killed_task,
            "in-flight tasks must be recorded as Failed(TaskKilled) after abort"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_long_delayed_failed_task_after_executor_lost() -> Result<()> {
        let executor1 = mock_executor("executor-id1".to_string());
        let executor2 = mock_executor("executor-id2".to_string());
        let mut agg_graph = test_aggregation_plan(4).await;
        // Call revive to move the leaf Resolved stages to Running
        agg_graph.revive();

        // Complete the Stage 1
        revive_graph_and_complete_next_stage_with_executor(&mut agg_graph, &executor1)?;

        // 1st task in the Stage 2
        if let Some(task) = agg_graph.pop_next_task(&executor2.id)? {
            let task_status = mock_completed_task(task, &executor2.id);
            agg_graph.update_task_status(&executor2, vec![task_status], 1, 1)?;
        }

        // 2rd task in the Stage 2
        if let Some(task) = agg_graph.pop_next_task(&executor1.id)? {
            let task_status = mock_completed_task(task, &executor1.id);
            agg_graph.update_task_status(&executor1, vec![task_status], 1, 1)?;
        }

        // 3rd task in the Stage 2, scheduled on executor 2 but not completed
        let task = agg_graph.pop_next_task(&executor2.id)?;

        // There is 1 task pending schedule now
        assert_eq!(agg_graph.available_tasks(), 1);

        // executor 1 lost
        let reset = agg_graph.reset_stages_on_lost_executor(&executor1.id)?;

        // Two stages were reset, Stage 2 rollback to Unresolved and Stage 1 move to Running
        assert_eq!(reset.0.len(), 2);
        assert_eq!(agg_graph.available_tasks(), 2);

        // Complete the Stage 1 again
        revive_graph_and_complete_next_stage_with_executor(&mut agg_graph, &executor1)?;

        // Stage 2 move to Running
        agg_graph.revive();
        assert_eq!(agg_graph.available_tasks(), 4);

        // 3rd task in Stage 2 update comes very late due to runtime execution error.
        let task_status = mock_failed_task(
            task.unwrap(),
            FailedTask {
                error: "ExecutionError".to_string(),
                retryable: false,
                count_to_failures: false,
                failed_reason: Some(failed_task::FailedReason::ExecutionError(
                    ExecutionError {},
                )),
            },
        );

        // This long delayed failed task should not failure the stage/job and should not trigger any query stage events
        let query_stage_events =
            agg_graph.update_task_status(&executor1, vec![task_status], 4, 4)?;
        assert!(query_stage_events.is_empty());

        drain_tasks(&mut agg_graph)?;
        assert!(agg_graph.is_successful(), "Failed to complete agg plan");

        Ok(())
    }

    #[tokio::test]
    async fn test_normal_fetch_failure() -> Result<()> {
        let executor1 = mock_executor("executor-id1".to_string());
        let executor2 = mock_executor("executor-id2".to_string());
        let mut agg_graph = test_aggregation_plan(4).await;
        // Call revive to move the leaf Resolved stages to Running
        agg_graph.revive();

        // Complete the Stage 1
        revive_graph_and_complete_next_stage(&mut agg_graph)?;

        // 1st task in the Stage 2
        let task1 = agg_graph.pop_next_task(&executor2.id)?.unwrap();
        let task_status1 = mock_completed_task(task1, &executor2.id);

        let task2 = agg_graph.pop_next_task(&executor2.id)?.unwrap();
        let failed_task = wrapped_fetch_failed_task(&executor1.id, 1, 0);
        assert!(matches!(
            failed_task.failed_reason,
            Some(failed_task::FailedReason::FetchPartitionError(_))
        ));
        let task_status2 = mock_failed_task(task2, failed_task);

        let mut running_task_count = 0;
        while let Some(_task) = agg_graph.pop_next_task(&executor2.id)? {
            running_task_count += 1;
        }
        assert_eq!(running_task_count, 2);

        let stage_events = agg_graph.update_task_status(
            &executor2,
            vec![task_status1, task_status2],
            4,
            4,
        )?;

        assert_eq!(stage_events.len(), 1);
        assert!(matches!(
            stage_events[0],
            QueryStageSchedulerEvent::CancelTasks(_)
        ));

        // Stage 1 is running
        let running_stage = agg_graph.running_stages();
        assert_eq!(running_stage.len(), 1);
        assert_eq!(running_stage[0], 1);
        assert_eq!(agg_graph.available_tasks(), 2);

        drain_tasks(&mut agg_graph)?;
        assert!(agg_graph.is_successful(), "Failed to complete agg plan");
        Ok(())
    }

    #[tokio::test]
    async fn test_many_fetch_failures_in_one_stage() -> Result<()> {
        let executor1 = mock_executor("executor-id1".to_string());
        let executor2 = mock_executor("executor-id2".to_string());
        let executor3 = mock_executor("executor-id3".to_string());
        let mut agg_graph = test_two_aggregations_plan(8).await;

        agg_graph.revive();
        assert_eq!(agg_graph.stage_count(), 3);

        // Complete the Stage 1
        revive_graph_and_complete_next_stage(&mut agg_graph)?;

        // Complete the Stage 2, 5 tasks run on executor_2 and 3 tasks run on executor_1
        for _i in 0..5 {
            if let Some(task) = agg_graph.pop_next_task(&executor2.id)? {
                let task_status = mock_completed_task(task, &executor2.id);
                agg_graph.update_task_status(&executor2, vec![task_status], 4, 4)?;
            }
        }
        assert_eq!(agg_graph.available_tasks(), 3);
        for _i in 0..3 {
            if let Some(task) = agg_graph.pop_next_task(&executor1.id)? {
                let task_status = mock_completed_task(task, &executor1.id);
                agg_graph.update_task_status(&executor1, vec![task_status], 4, 4)?;
            }
        }

        // Run Stage 3, 6 tasks failed due to FetchPartitionError on different map partitions on executor_2
        let mut many_fetch_failure_status = vec![];
        for part in 2..8 {
            if let Some(task) = agg_graph.pop_next_task(&executor3.id)? {
                let task_status = mock_failed_task(
                    task,
                    FailedTask {
                        error: "FetchPartitionError".to_string(),
                        retryable: false,
                        count_to_failures: false,
                        failed_reason: Some(
                            failed_task::FailedReason::FetchPartitionError(
                                FetchPartitionError {
                                    executor_id: executor2.id.clone(),
                                    map_stage_id: 2,
                                    map_partition_id: part,
                                },
                            ),
                        ),
                    },
                );
                many_fetch_failure_status.push(task_status);
            }
        }
        assert_eq!(many_fetch_failure_status.len(), 6);
        agg_graph.update_task_status(&executor3, many_fetch_failure_status, 4, 4)?;

        // The Running stage should be Stage 2 now
        let running_stage = agg_graph.running_stages();
        assert_eq!(running_stage.len(), 1);
        assert_eq!(running_stage[0], 2);
        assert_eq!(agg_graph.available_tasks(), 5);

        drain_tasks(&mut agg_graph)?;
        assert!(agg_graph.is_successful(), "Failed to complete agg plan");
        Ok(())
    }

    #[tokio::test]
    async fn test_many_consecutive_stage_fetch_failures() -> Result<()> {
        let executor1 = mock_executor("executor-id1".to_string());
        let executor2 = mock_executor("executor-id2".to_string());
        let mut agg_graph = test_aggregation_plan(4).await;
        // Call revive to move the leaf Resolved stages to Running
        agg_graph.revive();

        for attempt in 0..6 {
            revive_graph_and_complete_next_stage(&mut agg_graph)?;

            // 1rd task in the Stage 2, failed due to FetchPartitionError
            if let Some(task1) = agg_graph.pop_next_task(&executor2.id)? {
                let task_status1 = mock_failed_task(
                    task1.clone(),
                    FailedTask {
                        error: "FetchPartitionError".to_string(),
                        retryable: false,
                        count_to_failures: false,
                        failed_reason: Some(
                            failed_task::FailedReason::FetchPartitionError(
                                FetchPartitionError {
                                    executor_id: executor1.id.clone(),
                                    map_stage_id: 1,
                                    map_partition_id: 0,
                                },
                            ),
                        ),
                    },
                );

                let stage_events =
                    agg_graph.update_task_status(&executor2, vec![task_status1], 4, 4)?;

                if attempt < 3 {
                    // No JobRunningFailed stage events
                    assert_eq!(stage_events.len(), 0);
                    // Stage 1 is running
                    let running_stage = agg_graph.running_stages();
                    assert_eq!(running_stage.len(), 1);
                    assert_eq!(running_stage[0], 1);
                    assert_eq!(agg_graph.available_tasks(), 2);
                } else {
                    // Job is failed after exceeds the max_stage_failures
                    assert_eq!(stage_events.len(), 1);
                    assert!(matches!(
                        stage_events[0],
                        QueryStageSchedulerEvent::JobRunningFailed { .. }
                    ));
                    // Stage 2 is still running
                    let running_stage = agg_graph.running_stages();
                    assert_eq!(running_stage.len(), 1);
                    assert_eq!(running_stage[0], 2);
                }
            }
        }

        drain_tasks(&mut agg_graph)?;
        assert!(!agg_graph.is_successful(), "Expect to fail the agg plan");

        let failure_reason = format!("{:?}", agg_graph.status());
        assert!(failure_reason.contains("Job failed due to stage 2 failed: Stage 2 has failed 4 times, most recent failure reason"));
        assert!(failure_reason.contains("FetchPartitionError"));

        Ok(())
    }

    #[tokio::test]
    async fn test_long_delayed_fetch_failures() -> Result<()> {
        let executor1 = mock_executor("executor-id1".to_string());
        let executor2 = mock_executor("executor-id2".to_string());
        let executor3 = mock_executor("executor-id3".to_string());
        let mut agg_graph = test_two_aggregations_plan(8).await;

        agg_graph.revive();
        assert_eq!(agg_graph.stage_count(), 3);

        // Complete the Stage 1
        revive_graph_and_complete_next_stage(&mut agg_graph)?;

        // Complete the Stage 2, 5 tasks run on executor_2, 2 tasks run on executor_1, 1 task runs on executor_3
        for _i in 0..5 {
            if let Some(task) = agg_graph.pop_next_task(&executor2.id)? {
                let task_status = mock_completed_task(task, &executor2.id);
                agg_graph.update_task_status(&executor2, vec![task_status], 4, 4)?;
            }
        }
        assert_eq!(agg_graph.available_tasks(), 3);

        for _i in 0..2 {
            if let Some(task) = agg_graph.pop_next_task(&executor1.id)? {
                let task_status = mock_completed_task(task, &executor1.id);
                agg_graph.update_task_status(&executor1, vec![task_status], 4, 4)?;
            }
        }

        if let Some(task) = agg_graph.pop_next_task(&executor3.id)? {
            let task_status = mock_completed_task(task, &executor3.id);
            agg_graph.update_task_status(&executor3, vec![task_status], 4, 4)?;
        }
        assert_eq!(agg_graph.available_tasks(), 0);

        //Run Stage 3
        // 1st task scheduled
        let task_1 = agg_graph.pop_next_task(&executor3.id)?.unwrap();
        // 2nd task scheduled
        let task_2 = agg_graph.pop_next_task(&executor3.id)?.unwrap();
        // 3rd task scheduled
        let task_3 = agg_graph.pop_next_task(&executor3.id)?.unwrap();
        // 4th task scheduled
        let task_4 = agg_graph.pop_next_task(&executor3.id)?.unwrap();
        // 5th task scheduled
        let task_5 = agg_graph.pop_next_task(&executor3.id)?.unwrap();

        // Stage 3, 1st task failed due to FetchPartitionError(executor2)
        let task_status_1 = mock_failed_task(
            task_1,
            FailedTask {
                error: "FetchPartitionError".to_string(),
                retryable: false,
                count_to_failures: false,
                failed_reason: Some(failed_task::FailedReason::FetchPartitionError(
                    FetchPartitionError {
                        executor_id: executor2.id.clone(),
                        map_stage_id: 2,
                        map_partition_id: 0,
                    },
                )),
            },
        );
        agg_graph.update_task_status(&executor3, vec![task_status_1], 4, 4)?;

        // The Running stage is Stage 2 now
        let running_stage = agg_graph.running_stages();
        assert_eq!(running_stage.len(), 1);
        assert_eq!(running_stage[0], 2);
        assert_eq!(agg_graph.available_tasks(), 5);

        // Stage 3, 2nd task failed due to FetchPartitionError(executor2)
        let task_status_2 = mock_failed_task(
            task_2,
            FailedTask {
                error: "FetchPartitionError".to_string(),
                retryable: false,
                count_to_failures: false,
                failed_reason: Some(failed_task::FailedReason::FetchPartitionError(
                    FetchPartitionError {
                        executor_id: executor2.id.clone(),
                        map_stage_id: 2,
                        map_partition_id: 1,
                    },
                )),
            },
        );
        // This task update should be ignored
        agg_graph.update_task_status(&executor3, vec![task_status_2], 4, 4)?;
        let running_stage = agg_graph.running_stages();
        assert_eq!(running_stage.len(), 1);
        assert_eq!(running_stage[0], 2);
        assert_eq!(agg_graph.available_tasks(), 5);

        // Stage 3, 3rd task failed due to FetchPartitionError(executor1)
        let task_status_3 = mock_failed_task(
            task_3,
            FailedTask {
                error: "FetchPartitionError".to_string(),
                retryable: false,
                count_to_failures: false,
                failed_reason: Some(failed_task::FailedReason::FetchPartitionError(
                    FetchPartitionError {
                        executor_id: executor1.id.clone(),
                        map_stage_id: 2,
                        map_partition_id: 1,
                    },
                )),
            },
        );
        // This task update should be handled because it has a different failure reason
        agg_graph.update_task_status(&executor3, vec![task_status_3], 4, 4)?;
        // Running stage is still Stage 2, but available tasks changed to 7
        assert_eq!(running_stage.len(), 1);
        assert_eq!(running_stage[0], 2);
        assert_eq!(agg_graph.available_tasks(), 7);

        // Finish 4 tasks in Stage 2, to make some progress
        for _i in 0..4 {
            if let Some(task) = agg_graph.pop_next_task(&executor1.id)? {
                let task_status = mock_completed_task(task, &executor1.id);
                agg_graph.update_task_status(&executor1, vec![task_status], 4, 4)?;
            }
        }
        assert_eq!(running_stage.len(), 1);
        assert_eq!(running_stage[0], 2);
        assert_eq!(agg_graph.available_tasks(), 3);

        // Stage 3, 4th task failed due to FetchPartitionError(executor1)
        let task_status_4 = mock_failed_task(
            task_4,
            FailedTask {
                error: "FetchPartitionError".to_string(),
                retryable: false,
                count_to_failures: false,
                failed_reason: Some(failed_task::FailedReason::FetchPartitionError(
                    FetchPartitionError {
                        executor_id: executor1.id.clone(),
                        map_stage_id: 2,
                        map_partition_id: 1,
                    },
                )),
            },
        );
        // This task update should be ignored because the same failure reason is already handled
        agg_graph.update_task_status(&executor3, vec![task_status_4], 4, 4)?;
        let running_stage = agg_graph.running_stages();
        assert_eq!(running_stage.len(), 1);
        assert_eq!(running_stage[0], 2);
        assert_eq!(agg_graph.available_tasks(), 3);

        // Finish the other 3 tasks in Stage 2
        for _i in 0..3 {
            if let Some(task) = agg_graph.pop_next_task(&executor1.id)? {
                let task_status = mock_completed_task(task, &executor1.id);
                agg_graph.update_task_status(&executor1, vec![task_status], 4, 4)?;
            }
        }
        assert_eq!(agg_graph.available_tasks(), 0);

        // Stage 3, the very long delayed 5th task failed due to FetchPartitionError(executor3)
        // Although the failure reason is new, but this task should be ignored
        // Because its map stage's new attempt is finished and this stage's new attempt is running
        let task_status_5 = mock_failed_task(
            task_5,
            FailedTask {
                error: "FetchPartitionError".to_string(),
                retryable: false,
                count_to_failures: false,
                failed_reason: Some(failed_task::FailedReason::FetchPartitionError(
                    FetchPartitionError {
                        executor_id: executor3.id.clone(),
                        map_stage_id: 2,
                        map_partition_id: 1,
                    },
                )),
            },
        );
        agg_graph.update_task_status(&executor3, vec![task_status_5], 4, 4)?;
        // Stage 3's new attempt is running
        let running_stage = agg_graph.running_stages();
        assert_eq!(running_stage.len(), 1);
        assert_eq!(running_stage[0], 3);
        assert_eq!(agg_graph.available_tasks(), 8);

        // There is one failed stage attempts: Stage 3. Stage 2 does not count to failed attempts
        assert_eq!(agg_graph.failed_stage_attempts.len(), 1);
        assert_eq!(
            agg_graph.failed_stage_attempts.get(&3).cloned(),
            Some(HashSet::from([0]))
        );
        drain_tasks(&mut agg_graph)?;
        assert!(agg_graph.is_successful(), "Failed to complete agg plan");
        // Failed stage attempts are cleaned
        assert_eq!(agg_graph.failed_stage_attempts.len(), 0);

        Ok(())
    }

    #[tokio::test]
    // This test case covers a race condition in delayed fetch failure handling:
    // TaskStatus of input stage's new attempt come together with the parent stage's delayed FetchFailure
    async fn test_long_delayed_fetch_failures_race_condition() -> Result<()> {
        let executor1 = mock_executor("executor-id1".to_string());
        let executor2 = mock_executor("executor-id2".to_string());
        let executor3 = mock_executor("executor-id3".to_string());
        let mut agg_graph = test_two_aggregations_plan(8).await;

        agg_graph.revive();
        assert_eq!(agg_graph.stage_count(), 3);

        // Complete the Stage 1
        revive_graph_and_complete_next_stage(&mut agg_graph)?;

        // Complete the Stage 2, 5 tasks run on executor_2, 3 tasks run on executor_1
        for _i in 0..5 {
            if let Some(task) = agg_graph.pop_next_task(&executor2.id)? {
                let task_status = mock_completed_task(task, &executor2.id);
                agg_graph.update_task_status(&executor2, vec![task_status], 4, 4)?;
            }
        }
        assert_eq!(agg_graph.available_tasks(), 3);

        for _i in 0..3 {
            if let Some(task) = agg_graph.pop_next_task(&executor1.id)? {
                let task_status = mock_completed_task(task, &executor1.id);
                agg_graph.update_task_status(&executor1, vec![task_status], 4, 4)?;
            }
        }
        assert_eq!(agg_graph.available_tasks(), 0);

        // Run Stage 3
        // 1st task scheduled
        let task_1 = agg_graph.pop_next_task(&executor3.id)?.unwrap();
        // 2nd task scheduled
        let task_2 = agg_graph.pop_next_task(&executor3.id)?.unwrap();

        // Stage 3, 1st task failed due to FetchPartitionError(executor2)
        let task_status_1 = mock_failed_task(
            task_1,
            FailedTask {
                error: "FetchPartitionError".to_string(),
                retryable: false,
                count_to_failures: false,
                failed_reason: Some(failed_task::FailedReason::FetchPartitionError(
                    FetchPartitionError {
                        executor_id: executor2.id.clone(),
                        map_stage_id: 2,
                        map_partition_id: 0,
                    },
                )),
            },
        );
        agg_graph.update_task_status(&executor3, vec![task_status_1], 4, 4)?;

        // The Running stage is Stage 2 now
        let running_stage = agg_graph.running_stages();
        assert_eq!(running_stage.len(), 1);
        assert_eq!(running_stage[0], 2);
        assert_eq!(agg_graph.available_tasks(), 5);

        // Complete the 5 tasks in Stage 2's new attempts
        let mut task_status_vec = vec![];
        for _i in 0..5 {
            if let Some(task) = agg_graph.pop_next_task(&executor1.id)? {
                task_status_vec.push(mock_completed_task(task, &executor1.id))
            }
        }

        // Stage 3, 2nd task failed due to FetchPartitionError(executor1)
        let task_status_2 = mock_failed_task(
            task_2,
            FailedTask {
                error: "FetchPartitionError".to_string(),
                retryable: false,
                count_to_failures: false,
                failed_reason: Some(failed_task::FailedReason::FetchPartitionError(
                    FetchPartitionError {
                        executor_id: executor1.id.clone(),
                        map_stage_id: 2,
                        map_partition_id: 1,
                    },
                )),
            },
        );
        task_status_vec.push(task_status_2);

        // TaskStatus of Stage 2 come together with Stage 3 delayed FetchFailure update.
        // The successful tasks from Stage 2 would try to succeed the Stage2 and the delayed fetch failure try to reset the TaskInfo
        agg_graph.update_task_status(&executor3, task_status_vec, 4, 4)?;
        //The Running stage is still Stage 2, 3 new pending tasks added due to FetchPartitionError(executor1)
        assert_eq!(running_stage.len(), 1);
        assert_eq!(running_stage[0], 2);
        assert_eq!(agg_graph.available_tasks(), 3);

        drain_tasks(&mut agg_graph)?;
        assert!(agg_graph.is_successful(), "Failed to complete agg plan");

        Ok(())
    }

    #[tokio::test]
    async fn test_fetch_failures_in_different_stages() -> Result<()> {
        let executor1 = mock_executor("executor-id1".to_string());
        let executor2 = mock_executor("executor-id2".to_string());
        let executor3 = mock_executor("executor-id3".to_string());
        let mut agg_graph = test_two_aggregations_plan(8).await;

        agg_graph.revive();
        assert_eq!(agg_graph.stage_count(), 3);

        // Complete the Stage 1
        revive_graph_and_complete_next_stage(&mut agg_graph)?;

        // Complete the Stage 2, 5 tasks run on executor_2, 3 tasks run on executor_1
        for _i in 0..5 {
            if let Some(task) = agg_graph.pop_next_task(&executor2.id)? {
                let task_status = mock_completed_task(task, &executor2.id);
                agg_graph.update_task_status(&executor2, vec![task_status], 4, 4)?;
            }
        }
        assert_eq!(agg_graph.available_tasks(), 3);
        for _i in 0..3 {
            if let Some(task) = agg_graph.pop_next_task(&executor1.id)? {
                let task_status = mock_completed_task(task, &executor1.id);
                agg_graph.update_task_status(&executor1, vec![task_status], 4, 4)?;
            }
        }
        assert_eq!(agg_graph.available_tasks(), 0);

        // Run Stage 3
        // 1rd task in the Stage 3, failed due to FetchPartitionError(executor1)
        if let Some(task1) = agg_graph.pop_next_task(&executor3.id)? {
            let task_status1 = mock_failed_task(
                task1,
                FailedTask {
                    error: "FetchPartitionError".to_string(),
                    retryable: false,
                    count_to_failures: false,
                    failed_reason: Some(failed_task::FailedReason::FetchPartitionError(
                        FetchPartitionError {
                            executor_id: executor1.id.clone(),
                            map_stage_id: 2,
                            map_partition_id: 0,
                        },
                    )),
                },
            );

            let _stage_events =
                agg_graph.update_task_status(&executor3, vec![task_status1], 4, 4)?;
        }
        // The Running stage is Stage 2 now
        let running_stage = agg_graph.running_stages();
        assert_eq!(running_stage.len(), 1);
        assert_eq!(running_stage[0], 2);
        assert_eq!(agg_graph.available_tasks(), 3);

        // 1rd task in the Stage 2's new attempt, failed due to FetchPartitionError(executor1)
        if let Some(task1) = agg_graph.pop_next_task(&executor3.id)? {
            let task_status1 = mock_failed_task(
                task1,
                FailedTask {
                    error: "FetchPartitionError".to_string(),
                    retryable: false,
                    count_to_failures: false,
                    failed_reason: Some(failed_task::FailedReason::FetchPartitionError(
                        FetchPartitionError {
                            executor_id: executor1.id.clone(),
                            map_stage_id: 1,
                            map_partition_id: 0,
                        },
                    )),
                },
            );
            let _stage_events =
                agg_graph.update_task_status(&executor3, vec![task_status1], 4, 4)?;
        }
        // The Running stage is Stage 1 now
        let running_stage = agg_graph.running_stages();
        assert_eq!(running_stage.len(), 1);
        assert_eq!(running_stage[0], 1);
        assert_eq!(agg_graph.available_tasks(), 2);

        // There are two failed stage attempts: Stage 2 and Stage 3
        assert_eq!(agg_graph.failed_stage_attempts.len(), 2);
        assert_eq!(
            agg_graph.failed_stage_attempts.get(&2).cloned(),
            Some(HashSet::from([1]))
        );
        assert_eq!(
            agg_graph.failed_stage_attempts.get(&3).cloned(),
            Some(HashSet::from([0]))
        );

        drain_tasks(&mut agg_graph)?;
        assert!(agg_graph.is_successful(), "Failed to complete agg plan");
        assert_eq!(agg_graph.failed_stage_attempts.len(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn test_fetch_failure_with_normal_task_failure() -> Result<()> {
        let executor1 = mock_executor("executor-id1".to_string());
        let executor2 = mock_executor("executor-id2".to_string());
        let mut agg_graph = test_aggregation_plan(4).await;

        // Complete the Stage 1
        revive_graph_and_complete_next_stage(&mut agg_graph)?;

        // 1st task in the Stage 2
        let task1 = agg_graph.pop_next_task(&executor2.id)?.unwrap();
        let task_status1 = mock_completed_task(task1, &executor2.id);

        // 2nd task in the Stage 2, failed due to FetchPartitionError
        let task2 = agg_graph.pop_next_task(&executor2.id)?.unwrap();
        let task_status2 = mock_failed_task(
            task2,
            FailedTask {
                error: "FetchPartitionError".to_string(),
                retryable: false,
                count_to_failures: false,
                failed_reason: Some(failed_task::FailedReason::FetchPartitionError(
                    FetchPartitionError {
                        executor_id: executor1.id.clone(),
                        map_stage_id: 1,
                        map_partition_id: 0,
                    },
                )),
            },
        );

        // 3rd task in the Stage 2, failed due to ExecutionError
        let task3 = agg_graph.pop_next_task(&executor2.id)?.unwrap();
        let task_status3 = mock_failed_task(
            task3,
            FailedTask {
                error: "ExecutionError".to_string(),
                retryable: false,
                count_to_failures: false,
                failed_reason: Some(failed_task::FailedReason::ExecutionError(
                    ExecutionError {},
                )),
            },
        );

        let stage_events = agg_graph.update_task_status(
            &executor2,
            vec![task_status1, task_status2, task_status3],
            4,
            4,
        )?;

        assert_eq!(stage_events.len(), 1);
        assert!(matches!(
            stage_events[0],
            QueryStageSchedulerEvent::JobRunningFailed { .. }
        ));

        drain_tasks(&mut agg_graph)?;
        assert!(!agg_graph.is_successful(), "Expect to fail the agg plan");

        let failure_reason = format!("{:?}", agg_graph.status);
        assert!(failure_reason.contains("Job failed due to stage 2 failed"));
        assert!(failure_reason.contains("ExecutionError"));

        Ok(())
    }

    // #[tokio::test]
    // async fn test_shuffle_files_should_cleaned_after_fetch_failure() -> Result<()> {
    //     todo!()
    // }

    fn wrapped_fetch_failed_task(
        executor_id: &str,
        map_stage_id: usize,
        map_partition_id: usize,
    ) -> FailedTask {
        let err = BallistaError::DataFusionError(Box::new(
            BallistaError::FetchFailed(
                executor_id.to_owned(),
                map_stage_id,
                map_partition_id,
                "FetchPartitionError".to_owned(),
            )
            .into_datafusion(),
        ));
        FailedTask::from(err)
    }

    fn drain_tasks(graph: &mut dyn ExecutionGraph) -> Result<()> {
        let executor = mock_executor("executor-id1".to_string());
        while let Some(task) = graph.pop_next_task(&executor.id)? {
            let task_status = mock_completed_task(task, &executor.id);
            graph.update_task_status(&executor, vec![task_status], 1, 1)?;
        }

        Ok(())
    }
}
