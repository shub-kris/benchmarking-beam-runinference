PROJECT_ID = "apache-beam-testing"
# Subscription for PubSub Topic
SUBSCRIPTION_ID = f"projects/{PROJECT_ID}/subscriptions/newsgroup-dataset-subscription"
JOB_NAME = "benchmarking-runinference-cpu"
NUM_WORKERS = 1


TOKENIZER_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_STATE_DICT_PATH = (
    f"gs://{PROJECT_ID}-ml-examples/{TOKENIZER_NAME}/pytorch_model.bin"
)
MODEL_CONFIG_PATH = TOKENIZER_NAME
