#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments

from definitions import BERT_MODEL_NAME
from definitions import MAX_LEN
from src.utils.AMR import AMR


def encode_cat(x):
    if x not in encode_dict.keys():
        encode_dict[x]=x-1
    return encode_dict[x]

encode_dict = {}
df = pd.read_csv('./data/training.csv')
df = df.dropna()
df['overall_enc'] = df['overall'].apply(lambda x: encode_cat(x))

cfg = AutoConfig.from_pretrained(BERT_MODEL_NAME)
cfg.num_labels = df.overall.nunique()
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME, config=cfg)


train_idx, test_idx= train_test_split(
    np.arange(len(df)),
    test_size=0.15,
    shuffle=True,
    stratify=df['overall_enc'],
    random_state=1
)

all_data = AMR(df, tokenizer, MAX_LEN)
train_dataset = Subset(all_data, train_idx)
val_dataset = Subset(all_data, test_idx)

def compute_metrics(p):
    predictions, true_labels = p
    true_predictions = np.argmax(predictions, axis=-1)
    results = classification_report(
        true_labels, true_predictions, output_dict=True
    )
    print(results)
    return {
        "precision": results["macro avg"]["precision"],
        "recall": results["macro avg"]["recall"],
        "f1": results["macro avg"]["f1-score"],
    }

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=5,              # total number of training epochs
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=32,   # batch size for evaluation
    #     warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=64,
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    greater_is_better=True,
    metric_for_best_model='eval_f1',
    learning_rate=2e-5,
)

model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME, config=cfg)
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(f'./artifacts/{BERT_MODEL_NAME}')
tokenizer.save_pretrained(f'./artifacts/{BERT_MODEL_NAME}')
