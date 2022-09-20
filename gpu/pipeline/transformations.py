import apache_beam as beam
import torch
from apache_beam.io.filesystems import FileSystems
from apache_beam.ml.inference.pytorch_inference import PytorchModelHandlerKeyedTensor
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer


class CustomPytorchModelHandlerKeyedTensor(PytorchModelHandlerKeyedTensor):
    def load_model(self) -> torch.nn.Module:
        """Loads and initializes a Pytorch model for processing."""
        model = self._model_class(**self._model_params)
        model.to(self._device)
        file = FileSystems.open(self._state_dict_path, "rb")
        model.load_state_dict(torch.load(file, map_location=self._device))
        model.eval()
        return model


class HuggingFaceStripBatchingWrapper(DistilBertForSequenceClassification):
    """Wrapper around HugginFace model because RunInference requires a batch
    as a list of dicts instead of a dict of lists. Another workaround can be found
    here where they disable batching instead.
    https://github.com/apache/beam/blob/master/sdks/python/apache_beam/examples/inference/pytorch_language_modeling.py"""

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        return [dict(zip(output, v)) for v in zip(*output.values())]


class Tokenize(beam.DoFn):
    def __init__(self, model_name: str):
        self._model_name = model_name

    def setup(self):
        self._tokenizer = DistilBertTokenizer.from_pretrained(self._model_name)

    def process(self, text_input: str):
        # FRICTION POINT: We need to pad the tokens tensors to max length to make sure that all the tensors
        # are of the same length and hence stack-able by the RunInference API, normally you would batch first
        # and tokenize the batch after and pad each tensor the the max length in the batch.
        tokens = self._tokenizer(
            text_input, return_tensors="pt", padding="max_length", max_length=512
        )
        # squeeze because tokenization add an extra dimension, which is empty
        # in this case because we're tokenizing one element at a time.
        tokens = {key: torch.squeeze(val) for key, val in tokens.items()}
        # for key, val in tokens.items():
        #   print(f"{key}: {val.shape}")
        return [(text_input, tokens)]


class PostProcessor(beam.DoFn):
    def process(self, tuple_):
        text_input, prediction_result = tuple_
        softmax = (
            torch.nn.Softmax(dim=-1)(prediction_result.inference["logits"])
            .detach()
            .numpy()
        )
        return [{"input": text_input, "softmax": softmax}]
