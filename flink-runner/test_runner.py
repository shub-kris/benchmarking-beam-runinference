import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

options = PipelineOptions([
    "--runner=FlinkRunner",
    "--flink_version=1.14",
    "--flink_master=localhost:8081",
    "--environment_type=LOOPBACK"
])

with beam.Pipeline(
    options=options
    ) as pipeline:
  _ = (pipeline | "Create inputs" >> beam.Create(["test 1", "test 2"])
                | "Print" >> beam.Map(lambda x: print(x)))