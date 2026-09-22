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

#![cfg(feature = "standalone")]

mod common;

use ballista::prelude::SessionConfigExt;
use ballista_core::client::BallistaClient;
use ballista_core::config::{
    BALLISTA_SCHEDULER_MAX_PARTITIONS_PER_TASK,
    BALLISTA_SHUFFLE_READER_FORCE_REMOTE_READ,
    BALLISTA_SHUFFLE_READER_REMOTE_PREFER_FLIGHT, BallistaConfig,
};
use ballista_core::execution_plans::ChaosExec;
use ballista_core::execution_plans::pipelined_shuffle_reader::producing_input_observed;
use ballista_core::serde::BallistaCodec;
use ballista_core::serde::protobuf::{
    ExecuteQueryParams, ExecutorStoppedParams, GetJobMetricsParams, GetJobStatusParams,
    KeyValuePair, OpenShuffleInputParams, ShuffleInputHandle, execute_query_params,
    execute_query_result, job_status, operator_metric,
    scheduler_grpc_client::SchedulerGrpcClient, shuffle_input_result,
};
use ballista_core::serde::scheduler::ShuffleLayout;
use datafusion::arrow::array::Int32Array;
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::union::UnionExec;
use datafusion::physical_plan::{ExecutionPlan, Partitioning};
use datafusion::prelude::SessionConfig;
use datafusion_proto::physical_plan::AsExecutionPlan;
use datafusion_proto::protobuf::PhysicalPlanNode;
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

static E2E_LOCK: LazyLock<tokio::sync::Mutex<()>> =
    LazyLock::new(|| tokio::sync::Mutex::new(()));

async fn setup_cluster() -> (String, u16) {
    let address = ballista_scheduler::standalone::new_standalone_scheduler()
        .await
        .unwrap();
    let host = "localhost".to_owned();
    let scheduler =
        SchedulerGrpcClient::connect(format!("http://{host}:{}", address.port()))
            .await
            .unwrap();
    // Overlap requires a free slot beside the straggler, even on a single-core
    // runner. Keep the same explicit capacity for baseline and pipelined jobs.
    ballista_executor::new_standalone_executor(scheduler, 4, BallistaCodec::default())
        .await
        .unwrap();
    (host, address.port())
}

fn straggler_plan(delay_ms: u64) -> Arc<dyn ExecutionPlan> {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let mut inputs = vec![];
    for partition in 0..8 {
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![partition]))],
        )
        .unwrap();
        let input =
            MemorySourceConfig::try_new_exec(&[vec![batch]], schema.clone(), None)
                .unwrap();
        let input: Arc<dyn ExecutionPlan> = if partition == 7 {
            Arc::new(
                ChaosExec::new(input, 1.0, &format!("delay:{delay_ms}"), Some(7))
                    .unwrap(),
            )
        } else {
            input
        };
        inputs.push(input);
    }
    let input = UnionExec::try_new(inputs).unwrap();
    let partitioning = Partitioning::Hash(vec![Arc::new(Column::new("id", 0))], 4);
    let first: Arc<dyn ExecutionPlan> =
        Arc::new(RepartitionExec::try_new(input, partitioning.clone()).unwrap());
    let consumer: Arc<dyn ExecutionPlan> =
        Arc::new(ChaosExec::new(first, 1.0, "delay:100", Some(11)).unwrap());
    Arc::new(RepartitionExec::try_new(consumer, partitioning).unwrap())
}

fn assert_expected_rows(result: &[RecordBatch]) {
    let mut values: Vec<i32> = result
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .iter()
                .copied()
        })
        .collect();
    values.sort_unstable();
    assert_eq!(values, (0..8).collect::<Vec<_>>());
}

// Submit directly so assertions and failure injection use the scheduler's job
// ID. execute_physical_plan accepts a session ID, not a caller-chosen job ID.
async fn submit_query(
    scheduler: &mut SchedulerGrpcClient<tonic::transport::Channel>,
    config: &SessionConfig,
    delay_ms: u64,
) -> String {
    let codec = config.ballista_physical_extension_codec();
    let plan = PhysicalPlanNode::try_from_physical_plan(
        straggler_plan(delay_ms),
        codec.as_ref(),
    )
    .unwrap();
    let mut bytes = vec![];
    plan.try_encode(&mut bytes).unwrap();
    let result = scheduler
        .execute_query(ExecuteQueryParams {
            query: Some(execute_query_params::Query::PhysicalPlan(bytes)),
            session_id: String::new(),
            operation_id: uuid::Uuid::new_v4().to_string(),
            settings: config
                .options()
                .entries()
                .into_iter()
                .map(|entry| KeyValuePair {
                    key: entry.key,
                    value: entry.value,
                })
                .collect(),
        })
        .await
        .unwrap()
        .into_inner();
    match result.result.unwrap() {
        execute_query_result::Result::Success(success) => success.job_id,
        execute_query_result::Result::Failure(failure) => {
            panic!("query submission failed: {failure:?}")
        }
    }
}

async fn collect_job(
    mut scheduler: SchedulerGrpcClient<tonic::transport::Channel>,
    job_id: String,
) -> Vec<RecordBatch> {
    loop {
        let result = scheduler
            .get_job_status(GetJobStatusParams {
                job_id: job_id.clone(),
            })
            .await
            .unwrap()
            .into_inner();
        match result.status.and_then(|status| status.status) {
            Some(job_status::Status::Failed(failure)) => {
                panic!("job {job_id} failed: {}", failure.error)
            }
            Some(job_status::Status::Successful(success)) => {
                let mut batches = vec![];
                for location in success.partition_location {
                    let executor = location.executor_meta.unwrap();
                    let mut client = BallistaClient::try_new(
                        &executor.host,
                        executor.port as u16,
                        BallistaConfig::default().grpc_client_max_message_size(),
                        false,
                        None,
                        3,
                        100,
                        0,
                        0,
                    )
                    .await
                    .unwrap();
                    let stream = client
                        .fetch_partition(
                            &executor.id,
                            &location.partition_id.unwrap().into(),
                            location.file_id,
                            if location.is_sort_shuffle {
                                ShuffleLayout::Sort
                            } else {
                                ShuffleLayout::Passthrough
                            },
                            true,
                        )
                        .await
                        .unwrap();
                    batches.extend(
                        datafusion::physical_plan::common::collect(stream)
                            .await
                            .unwrap(),
                    );
                }
                return batches;
            }
            _ => tokio::time::sleep(Duration::from_millis(20)).await,
        }
    }
}

async fn run(enabled: bool, remote: bool, delay_ms: u64) -> Duration {
    let (host, port) = setup_cluster().await;
    let config = SessionConfig::new_with_ballista()
        .with_ballista_adaptive_query_planner(false)
        .with_ballista_shuffle_pipelined_enabled(enabled)
        .set_str(BALLISTA_SCHEDULER_MAX_PARTITIONS_PER_TASK, "1")
        .set_bool(BALLISTA_SHUFFLE_READER_FORCE_REMOTE_READ, remote)
        .set_bool(BALLISTA_SHUFFLE_READER_REMOTE_PREFER_FLIGHT, true);
    let mut scheduler = SchedulerGrpcClient::connect(format!("http://{host}:{port}"))
        .await
        .unwrap();
    let start = Instant::now();
    let job_id = submit_query(&mut scheduler, &config, delay_ms).await;
    let result = tokio::time::timeout(
        Duration::from_secs(60),
        collect_job(scheduler.clone(), job_id.clone()),
    )
    .await
    .expect("pipelined query must not deadlock");
    let duration = start.elapsed();
    assert_expected_rows(&result);

    if enabled {
        // A result-equivalence-only test also passes if admission silently
        // regresses to the full producer barrier. A reader that starts before
        // producer seal must observe at least one producing update and later a
        // sealed update, so one local reader partition necessarily polls twice.
        let metrics = scheduler
            .get_job_metrics(GetJobMetricsParams { job_id })
            .await
            .unwrap()
            .into_inner();
        let poll_counts: Vec<_> = metrics
            .stages
            .iter()
            .flat_map(|stage| stage.operators.iter())
            .filter(|operator| operator.operator_type == "PipelinedShuffleReaderExec")
            .flat_map(|operator| operator.metrics.iter())
            .filter_map(|metric| match (&metric.metric, metric.partition) {
                (Some(operator_metric::Metric::Count(count)), Some(_))
                    if count.name == "metadata_poll_count" =>
                {
                    Some(count.value)
                }
                _ => None,
            })
            .collect();
        assert!(
            !poll_counts.is_empty(),
            "pipelining enabled but no pipelined-reader poll metrics were reported"
        );
        assert!(
            poll_counts.iter().any(|count| *count > 1),
            "pipelining enabled but every reader started after producer seal: {poll_counts:?}"
        );
    }

    duration
}

#[tokio::test]
async fn straggler_results_match_with_local_and_flight_fetches() {
    let _guard = E2E_LOCK.lock().await;
    for remote in [false, true] {
        run(false, remote, 1000).await;
        run(true, remote, 1000).await;
    }
}

#[tokio::test]
async fn executor_loss_invalidates_live_generation_and_recovers_exactly_once() {
    let _guard = E2E_LOCK.lock().await;
    let (host, port) = setup_cluster().await;
    let mut scheduler = SchedulerGrpcClient::connect(format!("http://{host}:{port}"))
        .await
        .unwrap();
    let config = SessionConfig::new_with_ballista()
        .with_ballista_adaptive_query_planner(false)
        .with_ballista_shuffle_pipelined_enabled(true)
        .set_str(BALLISTA_SCHEDULER_MAX_PARTITIONS_PER_TASK, "1")
        .set_bool(BALLISTA_SHUFFLE_READER_FORCE_REMOTE_READ, true)
        .set_bool(BALLISTA_SHUFFLE_READER_REMOTE_PREFER_FLIGHT, true);
    let job_id = submit_query(&mut scheduler, &config, 5000).await;
    let query_job_id = job_id.clone();
    let query_scheduler = scheduler.clone();
    let query = tokio::spawn(async move {
        tokio::time::timeout(
            Duration::from_secs(90),
            collect_job(query_scheduler, query_job_id),
        )
        .await
        .expect("recovery query must not deadlock")
    });

    let output_partitions = vec![0, 1, 2, 3];
    let (stage_id, executor_id) = tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            for stage_id in 0..16_u32 {
                let handle = ShuffleInputHandle {
                    job_id: job_id.clone(),
                    stage_id,
                    generation: 1,
                };
                let Ok(response) = scheduler
                    .open_shuffle_input(OpenShuffleInputParams {
                        handle: Some(handle),
                        output_partition_ids: output_partitions.clone(),
                    })
                    .await
                else {
                    continue;
                };
                let Some(shuffle_input_result::Result::Update(update)) =
                    response.into_inner().result
                else {
                    continue;
                };
                if update.lifecycle != 0 || update.locations.is_empty() {
                    continue;
                }
                let executor_id = update
                    .locations
                    .iter()
                    .filter_map(|block| block.location.as_ref())
                    .filter_map(|location| location.executor_meta.as_ref())
                    .map(|executor| executor.id.clone())
                    .find(|id| !id.is_empty());
                if let Some(executor_id) = executor_id {
                    return (stage_id, executor_id);
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("must observe committed output from an unsealed producer");

    tokio::time::timeout(Duration::from_secs(10), async {
        while !producing_input_observed(&job_id, stage_id, 1) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the reader pinned to this producer generation must execute before seal");

    let old_handle = ShuffleInputHandle {
        job_id: job_id.clone(),
        stage_id,
        generation: 1,
    };
    let response = scheduler
        .open_shuffle_input(OpenShuffleInputParams {
            handle: Some(old_handle.clone()),
            output_partition_ids: output_partitions.clone(),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(
        matches!(
            response.result,
            Some(shuffle_input_result::Result::Update(ref update))
                if update.lifecycle == 0
        ),
        "consumer must be live while the producer generation is still unsealed"
    );

    scheduler
        .executor_stopped(ExecutorStoppedParams {
            executor_id,
            reason: "pipelined shuffle recovery integration test".into(),
        })
        .await
        .unwrap();

    let current_generation = tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            if let Ok(response) = scheduler
                .open_shuffle_input(OpenShuffleInputParams {
                    handle: Some(old_handle.clone()),
                    output_partition_ids: output_partitions.clone(),
                })
                .await
                && let Some(shuffle_input_result::Result::Invalidated(invalidated)) =
                    response.into_inner().result
            {
                return invalidated.current_generation;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("lost committed output must invalidate the pinned generation");
    assert!(
        current_generation > old_handle.generation,
        "recovery must roll the shuffle generation forward"
    );

    // The cluster starts with one executor. Once it is declared lost, provide
    // replacement capacity instead of depending on a dead executor rejoining.
    ballista_executor::new_standalone_executor(
        scheduler.clone(),
        4,
        BallistaCodec::default(),
    )
    .await
    .unwrap();

    let result = query.await.expect("recovery query task must not panic");
    assert_expected_rows(&result);
}

fn median_millis(samples: &[Duration]) -> u128 {
    let mut values: Vec<_> = samples.iter().map(Duration::as_millis).collect();
    values.sort_unstable();
    values[values.len() / 2]
}

#[tokio::test]
#[ignore = "diagnostic latency samples; run explicitly on an otherwise idle runner"]
async fn straggler_latency() {
    let _guard = E2E_LOCK.lock().await;
    const SAMPLES: usize = 3;

    for remote in [false, true] {
        let mut baseline = Vec::with_capacity(SAMPLES);
        let mut pipelined = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            baseline.push(run(false, remote, 1500).await);
            pipelined.push(run(true, remote, 1500).await);
        }
        println!(
            "remote={remote} baseline_ms={:?} baseline_median_ms={} pipelined_ms={:?} pipelined_median_ms={}",
            baseline.iter().map(Duration::as_millis).collect::<Vec<_>>(),
            median_millis(&baseline),
            pipelined
                .iter()
                .map(Duration::as_millis)
                .collect::<Vec<_>>(),
            median_millis(&pipelined),
        );
    }
}
