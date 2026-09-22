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

//! Incremental discovery over immutable materialized shuffle files.

use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{
    ExecutionPlanMetricsSet, MetricBuilder, MetricsSet,
};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream, Statistics,
};
use futures::TryStreamExt;
use tonic::codegen::{Body, Bytes, StdError};

use super::ShuffleReaderExec;
use crate::client_pool::BallistaClientPool;
use crate::error::BallistaError;
use crate::serde::protobuf::{self as pb, scheduler_grpc_client::SchedulerGrpcClient};
use crate::serde::scheduler::PartitionLocation;

/// Runtime-only discovery transport; physical plans serialize only the handle.
#[async_trait]
pub trait ShuffleInputClient: Debug + Send + Sync {
    async fn update(
        &self,
        handle: pb::ShuffleInputHandle,
        partition: u32,
        after: Option<u64>,
    ) -> std::result::Result<pb::ShuffleInputResult, tonic::Status>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::JobId;
    use crate::serde::scheduler::{ExecutorMetadata, PartitionId, PartitionStats};
    use datafusion::arrow::array::Int32Array;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::arrow::ipc::writer::StreamWriter;
    use datafusion::arrow::record_batch::RecordBatch;
    use datafusion::prelude::SessionConfig;
    use futures::StreamExt;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    #[derive(Debug)]
    struct ScriptedSource {
        responses: Mutex<VecDeque<pb::ShuffleInputResult>>,
        cursors: Mutex<Vec<Option<u64>>>,
    }

    impl ScriptedSource {
        fn new(responses: Vec<pb::ShuffleInputResult>) -> Arc<Self> {
            Arc::new(Self {
                responses: Mutex::new(responses.into()),
                cursors: Mutex::new(vec![]),
            })
        }

        fn cursors(&self) -> Vec<Option<u64>> {
            self.cursors.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl ShuffleInputClient for ScriptedSource {
        async fn update(
            &self,
            handle: pb::ShuffleInputHandle,
            partition: u32,
            after: Option<u64>,
        ) -> std::result::Result<pb::ShuffleInputResult, tonic::Status> {
            assert_eq!(handle.job_id, "job");
            assert_eq!(handle.stage_id, 1);
            assert_eq!(handle.generation, 1);
            assert_eq!(partition, 6);
            self.cursors.lock().unwrap().push(after);
            self.responses.lock().unwrap().pop_front().ok_or_else(|| {
                tonic::Status::internal("unexpected shuffle metadata poll")
            })
        }
    }

    fn metadata_update(
        version: u64,
        lifecycle: i32,
        locations: Vec<(u64, pb::PartitionLocation)>,
    ) -> pb::ShuffleInputResult {
        pb::ShuffleInputResult {
            result: Some(pb::shuffle_input_result::Result::Update(
                pb::ShuffleInputUpdate {
                    generation: 1,
                    version,
                    lifecycle,
                    locations: locations
                        .into_iter()
                        .map(|(published_version, location)| {
                            pb::VersionedPartitionLocation {
                                published_version,
                                location: Some(location),
                            }
                        })
                        .collect(),
                },
            )),
        }
    }

    fn invalidated(
        expected_generation: u64,
        current_generation: u64,
    ) -> pb::ShuffleInputResult {
        pb::ShuffleInputResult {
            result: Some(pb::shuffle_input_result::Result::Invalidated(
                pb::ShuffleGenerationInvalidated {
                    expected_generation,
                    current_generation,
                },
            )),
        }
    }

    fn test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            false,
        )]))
    }

    fn materialized_location(
        directory: &tempfile::TempDir,
        task: usize,
        file_id: u64,
        value: i32,
        schema: &SchemaRef,
    ) -> Result<pb::PartitionLocation> {
        let location = PartitionLocation {
            map_partition_id: task,
            partition_id: PartitionId::new(&JobId::from("job"), 1, 6),
            executor_meta: ExecutorMetadata {
                id: "local".into(),
                host: "localhost".into(),
                port: 1,
                grpc_port: 2,
                specification: Default::default(),
                os_info: Default::default(),
            },
            partition_stats: PartitionStats::default(),
            file_id: Some(file_id),
            is_sort_shuffle: false,
        };
        let path = location.path(directory.path().to_str().unwrap())?;
        std::fs::create_dir_all(path.parent().unwrap())?;
        let mut writer = StreamWriter::try_new(std::fs::File::create(path)?, schema)?;
        writer.write(&RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![value]))],
        )?)?;
        writer.finish()?;
        location
            .try_into()
            .map_err(|e: BallistaError| e.into_datafusion())
    }

    fn execute_script(
        source: Arc<dyn ShuffleInputClient>,
        directory: &tempfile::TempDir,
        schema: SchemaRef,
    ) -> Result<SendableRecordBatchStream> {
        let config =
            SessionConfig::new().with_extension(Arc::new(ShuffleInputRuntime(source)));
        let context = Arc::new(TaskContext::default().with_session_config(config));
        PipelinedShuffleReaderExec::try_new(
            pb::ShuffleInputHandle {
                job_id: "job".into(),
                stage_id: 1,
                generation: 1,
            },
            vec![6],
            schema,
            Partitioning::UnknownPartitioning(1),
        )?
        .with_fetch_runtime(directory.path().to_string_lossy().into_owned(), None)
        .execute(0, context)
    }

    #[tokio::test]
    async fn producing_empty_snapshot_waits_until_seal() -> Result<()> {
        let source = ScriptedSource::new(vec![
            metadata_update(0, 0, vec![]),
            metadata_update(1, 1, vec![]),
        ]);
        let directory = tempfile::tempdir()?;
        let mut stream =
            execute_script(source.clone(), &directory, Arc::new(Schema::empty()))?;
        assert!(stream.next().await.is_none());
        assert_eq!(source.cursors(), vec![None, Some(0)]);
        Ok(())
    }

    #[tokio::test]
    async fn partition_local_cursor_advances_across_empty_global_update() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let schema = test_schema();
        let location = materialized_location(&directory, 0, 0, 7, &schema)?;
        let source = ScriptedSource::new(vec![
            // Another output partition may advance the generation-wide version.
            metadata_update(1, 0, vec![]),
            metadata_update(2, 1, vec![(2, location)]),
        ]);
        let mut stream = execute_script(source.clone(), &directory, schema)?;
        let batch = stream
            .next()
            .await
            .transpose()?
            .expect("one materialized batch");
        let values = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(values.len(), 1);
        assert_eq!(values.value(0), 7);
        assert!(stream.next().await.is_none());
        assert_eq!(source.cursors(), vec![None, Some(1)]);
        Ok(())
    }

    #[tokio::test]
    async fn stale_delta_location_is_rejected() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let schema = test_schema();
        let location = materialized_location(&directory, 0, 0, 7, &schema)?;
        let source = ScriptedSource::new(vec![
            metadata_update(1, 0, vec![]),
            // A poll after v1 must never replay a v1 publication.
            metadata_update(2, 1, vec![(1, location)]),
        ]);
        let mut stream = execute_script(source, &directory, schema)?;
        let error = stream
            .next()
            .await
            .expect("protocol violation must fail")
            .expect_err("stale publication must fail closed");
        assert!(
            error
                .to_string()
                .contains("invalid shuffle publication version")
        );
        Ok(())
    }

    #[tokio::test]
    async fn seal_without_version_advance_is_rejected() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let source = ScriptedSource::new(vec![
            metadata_update(1, 0, vec![]),
            metadata_update(1, 1, vec![]),
        ]);
        let mut stream = execute_script(source, &directory, Arc::new(Schema::empty()))?;
        let error = stream
            .next()
            .await
            .expect("invalid seal must fail")
            .expect_err("seal must advance metadata history");
        assert!(
            error
                .to_string()
                .contains("invalid shuffle metadata history")
        );
        Ok(())
    }

    #[tokio::test]
    async fn regressing_update_version_is_rejected() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let source = ScriptedSource::new(vec![
            metadata_update(2, 0, vec![]),
            metadata_update(1, 1, vec![]),
        ]);
        let mut stream = execute_script(source, &directory, Arc::new(Schema::empty()))?;
        let error = stream
            .next()
            .await
            .expect("protocol violation must fail")
            .expect_err("regressing cursor must fail closed");
        assert!(
            error
                .to_string()
                .contains("invalid shuffle metadata history")
        );
        Ok(())
    }

    #[tokio::test]
    async fn exact_duplicate_in_snapshot_is_read_once() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let schema = test_schema();
        let location = materialized_location(&directory, 0, 0, 11, &schema)?;
        let source = ScriptedSource::new(vec![metadata_update(
            1,
            1,
            vec![(1, location.clone()), (1, location)],
        )]);
        let mut stream = execute_script(source, &directory, schema)?;
        let batch = stream
            .next()
            .await
            .transpose()?
            .expect("one materialized batch");
        let values = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(values.len(), 1);
        assert_eq!(values.value(0), 11);
        assert!(stream.next().await.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn distinct_file_ids_are_distinct_shuffle_artifacts() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let schema = test_schema();
        let first = materialized_location(&directory, 0, 0, 11, &schema)?;
        let second = materialized_location(&directory, 0, 99, 12, &schema)?;
        let source = ScriptedSource::new(vec![metadata_update(
            1,
            1,
            vec![(1, first), (1, second)],
        )]);
        let mut stream = execute_script(source, &directory, schema)?;
        let mut values = Vec::new();
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            values.extend(
                batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied(),
            );
        }
        values.sort_unstable();
        assert_eq!(values, vec![11, 12]);
        Ok(())
    }

    #[tokio::test]
    async fn invalid_invalidation_history_is_rejected() -> Result<()> {
        for response in [invalidated(2, 3), invalidated(1, 1)] {
            let directory = tempfile::tempdir()?;
            let source = ScriptedSource::new(vec![response]);
            let mut stream =
                execute_script(source, &directory, Arc::new(Schema::empty()))?;
            let error = stream
                .next()
                .await
                .expect("invalid invalidation must fail")
                .expect_err("invalid invalidation envelope must fail closed");
            assert!(
                error
                    .to_string()
                    .contains("invalid shuffle invalidation history")
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn invalidation_after_consumption_never_mixes_generations() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let schema = test_schema();
        let location = materialized_location(&directory, 0, 0, 13, &schema)?;
        let source = ScriptedSource::new(vec![
            metadata_update(1, 0, vec![(1, location)]),
            invalidated(1, 2),
        ]);
        let mut stream = execute_script(source.clone(), &directory, schema)?;
        let batch = stream
            .next()
            .await
            .transpose()?
            .expect("first generation batch");
        let values = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(values.len(), 1);
        assert_eq!(values.value(0), 13);
        let error = stream
            .next()
            .await
            .expect("invalidation must surface as an error")
            .expect_err("invalidation is never EOF");
        assert!(matches!(
            BallistaError::from(error),
            BallistaError::ShuffleGenerationInvalidated {
                stage_id: 1,
                expected: 1,
                current: 2,
            }
        ));
        assert_eq!(source.cursors(), vec![None, Some(1)]);
        Ok(())
    }
}

#[async_trait]
impl<C> ShuffleInputClient for SchedulerGrpcClient<C>
where
    C: tonic::client::GrpcService<tonic::body::Body>
        + Clone
        + Debug
        + Send
        + Sync
        + 'static,
    C::Error: Into<StdError>,
    C::Future: Send,
    C::ResponseBody: Body<Data = Bytes> + Send + 'static,
    <C::ResponseBody as Body>::Error: Into<StdError> + Send,
{
    async fn update(
        &self,
        handle: pb::ShuffleInputHandle,
        partition: u32,
        after: Option<u64>,
    ) -> std::result::Result<pb::ShuffleInputResult, tonic::Status> {
        let mut client = self.clone();
        let request = async {
            match after {
                None => {
                    client
                        .open_shuffle_input(pb::OpenShuffleInputParams {
                            handle: Some(handle),
                            output_partition_ids: vec![partition],
                        })
                        .await
                }
                Some(after_version) => {
                    client
                        .poll_shuffle_input(pb::PollShuffleInputParams {
                            handle: Some(handle),
                            output_partition_ids: vec![partition],
                            after_version,
                            max_wait_ms: 1000,
                        })
                        .await
                }
            }
        };
        tokio::time::timeout(Duration::from_secs(35), request)
            .await
            .map_err(|_| tonic::Status::deadline_exceeded("shuffle metadata timed out"))?
            .map(tonic::Response::into_inner)
    }
}

/// Injected into executor task sessions, never encoded in plan protobufs.
#[derive(Debug, Clone)]
pub struct ShuffleInputRuntime(pub Arc<dyn ShuffleInputClient>);

#[derive(Debug, Clone)]
pub struct PipelinedShuffleReaderExec {
    pub handle: pb::ShuffleInputHandle,
    pub upstream_partition_ids: Vec<usize>,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
    work_dir: Option<String>,
    client_pool: Option<Arc<dyn BallistaClientPool>>,
}

impl PipelinedShuffleReaderExec {
    pub fn try_new(
        handle: pb::ShuffleInputHandle,
        upstream_partition_ids: Vec<usize>,
        schema: SchemaRef,
        partitioning: Partitioning,
    ) -> Result<Self> {
        if handle.job_id.is_empty()
            || handle.generation == 0
            || upstream_partition_ids.is_empty()
            || upstream_partition_ids.len() != partitioning.partition_count()
            || upstream_partition_ids
                .iter()
                .any(|p| u32::try_from(*p).is_err())
            || upstream_partition_ids.iter().collect::<HashSet<_>>().len()
                != upstream_partition_ids.len()
        {
            return Err(DataFusionError::Plan(
                "invalid pipelined shuffle descriptor".into(),
            ));
        }
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            partitioning,
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Ok(Self {
            handle,
            upstream_partition_ids,
            schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
            work_dir: None,
            client_pool: None,
        })
    }

    pub fn with_fetch_runtime(
        &self,
        work_dir: String,
        client_pool: Option<Arc<dyn BallistaClientPool>>,
    ) -> Self {
        Self {
            work_dir: Some(work_dir),
            client_pool,
            ..self.clone()
        }
    }
}

impl DisplayAs for PipelinedShuffleReaderExec {
    fn fmt_as(&self, _: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        write!(
            f,
            "PipelinedShuffleReaderExec: stage={}, generation={}, partitions={:?}",
            self.handle.stage_id, self.handle.generation, self.upstream_partition_ids
        )
    }
}

impl ExecutionPlan for PipelinedShuffleReaderExec {
    fn name(&self) -> &str {
        "PipelinedShuffleReaderExec"
    }
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }
    fn apply_expressions(
        &self,
        _: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(DataFusionError::Plan(
                "shuffle reader has no children".into(),
            ));
        }
        Ok(self)
    }
    fn partition_statistics(&self, _: Option<usize>) -> Result<Arc<Statistics>> {
        // Partial committed metadata must never masquerade as final statistics.
        Ok(Arc::new(Statistics::new_unknown(&self.schema)))
    }
    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let upstream = *self.upstream_partition_ids.get(partition).ok_or_else(|| {
            DataFusionError::Execution("invalid local shuffle partition".into())
        })? as u32;
        let runtime = context
            .session_config()
            .get_extension::<ShuffleInputRuntime>()
            .ok_or_else(|| {
                DataFusionError::Configuration(
                    "executor missing shuffle metadata client".into(),
                )
            })?;
        let work_dir = self.work_dir.clone().ok_or_else(|| {
            DataFusionError::Configuration(
                "executor missing shuffle work directory".into(),
            )
        })?;
        let reader = self.clone();
        let discovered =
            MetricBuilder::new(&self.metrics).counter("locations_discovered", partition);
        let duplicates = MetricBuilder::new(&self.metrics)
            .counter("duplicate_locations_ignored", partition);
        let polls =
            MetricBuilder::new(&self.metrics).counter("metadata_poll_count", partition);
        let updates =
            MetricBuilder::new(&self.metrics).counter("metadata_updates", partition);
        let wait = MetricBuilder::new(&self.metrics)
            .subset_time("metadata_wait_time", partition);
        let invalidations = MetricBuilder::new(&self.metrics)
            .counter("generation_invalidations", partition);
        // A stream of finite static fetch streams reuses the existing local/Flight
        // engine, its concurrency limits, retries, and cancellation behavior.
        // Metadata discovery intentionally follows a fetch-then-poll cadence:
        // invalidation is observed between finite fetch streams, not concurrently
        // while one static batch is being drained.
        let stream = futures::stream::try_unfold(
            (None, HashMap::new(), false),
            move |(after, mut seen, sealed)| {
                let reader = reader.clone();
                let runtime = runtime.clone();
                let context = context.clone();
                let work_dir = work_dir.clone();
                let discovered = discovered.clone();
                let duplicates = duplicates.clone();
                let polls = polls.clone();
                let updates = updates.clone();
                let wait = wait.clone();
                let invalidations = invalidations.clone();
                async move {
                    if sealed {
                        return Ok(None);
                    }
                    let mut failures = 0;
                    let response = loop {
                        polls.add(1);
                        let timer = wait.timer();
                        let response = runtime
                            .0
                            .update(reader.handle.clone(), upstream, after)
                            .await;
                        drop(timer);
                        match response {
                            Ok(response) => break response,
                            Err(status)
                                if failures < 3
                                    && matches!(
                                        status.code(),
                                        tonic::Code::Unavailable
                                            | tonic::Code::DeadlineExceeded
                                    ) =>
                            {
                                failures += 1;
                                tokio::time::sleep(Duration::from_millis(100 * failures))
                                    .await;
                            }
                            Err(status) => {
                                return Err(BallistaError::from(status).into_datafusion());
                            }
                        }
                    };
                    let update = match response.result {
                        Some(pb::shuffle_input_result::Result::Update(update)) => update,
                        Some(pb::shuffle_input_result::Result::Invalidated(
                            invalidated,
                        )) => {
                            if invalidated.expected_generation != reader.handle.generation
                                || invalidated.current_generation
                                    <= invalidated.expected_generation
                            {
                                return Err(DataFusionError::Execution(
                                    "invalid shuffle invalidation history".into(),
                                ));
                            }
                            invalidations.add(1);
                            return Err(BallistaError::ShuffleGenerationInvalidated {
                                stage_id: reader.handle.stage_id as usize,
                                expected: reader.handle.generation,
                                current: invalidated.current_generation,
                            }
                            .into_datafusion());
                        }
                        None => {
                            return Err(DataFusionError::Execution(
                                "missing shuffle metadata result".into(),
                            ));
                        }
                    };
                    if update.generation != reader.handle.generation
                        || update.version < after.unwrap_or(0)
                        || !matches!(update.lifecycle, 0 | 1)
                        || (update.lifecycle == 1 && update.version == after.unwrap_or(0))
                    {
                        return Err(DataFusionError::Execution(
                            "invalid shuffle metadata history".into(),
                        ));
                    }
                    updates.add(1);
                    let mut locations = Vec::new();
                    for block in update.locations {
                        if block.published_version == 0
                            || block.published_version > update.version
                            || after
                                .is_some_and(|cursor| block.published_version <= cursor)
                        {
                            return Err(DataFusionError::Execution(
                                "invalid shuffle publication version".into(),
                            ));
                        }
                        let location: PartitionLocation = block
                            .location
                            .ok_or_else(|| {
                                DataFusionError::Execution(
                                    "missing shuffle location".into(),
                                )
                            })?
                            .try_into()
                            .map_err(|e: BallistaError| e.into_datafusion())?;
                        if location.partition_id.job_id.as_str() != reader.handle.job_id
                            || location.partition_id.stage_id
                                != reader.handle.stage_id as usize
                            || location.partition_id.partition_id != upstream as usize
                        {
                            return Err(DataFusionError::Execution(
                                "shuffle location identity mismatch".into(),
                            ));
                        }
                        let key = (
                            location.map_partition_id,
                            location.partition_id.partition_id,
                            location.file_id,
                        );
                        match seen.get(&key) {
                            None => {
                                seen.insert(key, location.clone());
                                locations.push(location);
                                discovered.add(1);
                            }
                            Some(previous) if previous == &location => {
                                duplicates.add(1);
                            }
                            Some(_) => {
                                return Err(DataFusionError::Execution(
                                    "conflicting shuffle location replay".into(),
                                ));
                            }
                        }
                    }
                    let stream: SendableRecordBatchStream = if locations.is_empty() {
                        Box::pin(RecordBatchStreamAdapter::new(
                            reader.schema.clone(),
                            futures::stream::empty::<
                                Result<datafusion::arrow::record_batch::RecordBatch>,
                            >(),
                        ))
                    } else {
                        let mut fetch = ShuffleReaderExec::try_new(
                            reader.handle.stage_id as usize,
                            vec![locations],
                            reader.schema.clone(),
                            Partitioning::UnknownPartitioning(1),
                        )?
                        .with_work_dir(work_dir);
                        if let Some(pool) = reader.client_pool {
                            fetch = fetch.with_client_pool(pool);
                        }
                        fetch.execute(0, context)?
                    };
                    Ok(Some((
                        stream,
                        (Some(update.version), seen, update.lifecycle == 1),
                    )))
                }
            },
        )
        .try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema.clone(),
            stream,
        )))
    }
}
