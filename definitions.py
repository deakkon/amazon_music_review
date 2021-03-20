import os

AMR_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/"
# BERT_MODEL_NAME = 'distilbert-base-cased'
BERT_MODEL_NAME = 'activebus/BERT_Review'
# BERT_MODEL_NAME = 'allenai/reviews_roberta_base'
BERT_ARTIFACTS = f'{AMR_ROOT}artifacts/{BERT_MODEL_NAME}/'
MAX_LEN = 256
