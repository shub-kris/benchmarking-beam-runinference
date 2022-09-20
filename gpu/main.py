import argparse
import sys

import apache_beam as beam
from apache_beam.ml.inference import RunInference
from apache_beam.ml.inference.base import KeyedModelHandler
from apache_beam.ml.inference.pytorch_inference import PytorchModelHandlerKeyedTensor
from transformers import DistilBertConfig

import config as cfg
from pipeline.options import get_pipeline_options
from pipeline.transformations import (
    CustomPytorchModelHandlerKeyedTensor,
    HuggingFaceStripBatchingWrapper,
    PostProcessor,
    Tokenize,
)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="benchmark-runinference")

    parser.add_argument(
        "-m",
        "--mode",
        help="Mode to run pipeline in.",
        choices=["local", "cloud"],
        default="local",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="GCP project to run pipeline on.",
        default=cfg.PROJECT_ID,
    )

    args, _ = parser.parse_known_args(args=argv)
    return args


def run():
    args = parse_arguments(sys.argv)

    inputs = [
        "This is the worst food I have ever eaten",
        "In my soul and in my heart, I’m convinced I’m wrong!",
        "Be with me always—take any form—drive me mad! only do not leave me in this abyss, where I cannot find you!",
        "Do I want to live? Would you like to live with your soul in the grave?",
        "Honest people don’t hide their deeds.",
        "Nelly, I am Heathcliff!  He’s always, always in my mind: not as a pleasure, any more than I am always a pleasure to myself, but as my own being.",
    ]

    pipeline_options = get_pipeline_options(
        job_name=cfg.JOB_NAME,
        num_workers=cfg.NUM_WORKERS,
        project=args.project,
        mode=args.mode,
    )

    # model_handler = PytorchModelHandlerKeyedTensor(
    #     state_dict_path=cfg.MODEL_STATE_DICT_PATH,
    #     model_class=HuggingFaceStripBatchingWrapper,
    #     model_params={
    #         "config": DistilBertConfig.from_pretrained(cfg.MODEL_CONFIG_PATH)
    #     },
    #     device="cuda:0",
    # )

    with beam.Pipeline(options=pipeline_options) as pipeline:
        _ = (
            pipeline
            | "Create inputs" >> beam.Create(inputs)
            # | "Tokenize" >> beam.ParDo(Tokenize(cfg.TOKENIZER_NAME))
            # | "Inference"
            # >> RunInference(model_handler=KeyedModelHandler(model_handler))
            # | "Decode Predictions" >> beam.ParDo(PostProcessor())
            # | "Print"
            # >> beam.Map(
            # lambda x: print(
            # f"Input: {x['input']} -> negative={100 * x['softmax'][0]:.4f}%/positive={100 * x['softmax'][1]:.4f}%"
            # )
            # )
            | "Print" >> beam.Map(lambda x: print(x))
        )
    # metrics = pipeline.result.metrics().query(beam.metrics.MetricsFilter())
    # print(metrics)


if __name__ == "__main__":
    run()
