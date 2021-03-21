#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from joblib import dump
from joblib import load
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Subset
from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments

from definitions import AMR_ROOT
from definitions import BERT_ARTIFACTS
from definitions import BERT_MODEL_NAME
from definitions import MAX_LEN
from src.utils.AMR import AMR


class ModelTrainer:

    def __init__(self, filename: str=None, serialization_path=None):
        self.filename = filename if filename else f'{AMR_ROOT}data/training.csv'
        self.serialization_path = serialization_path if serialization_path else BERT_ARTIFACTS
        if not self.serialization_path[-1] == "/":
            self.serialization_path = f"{self.serialization_path}/"
        self.df = pd.read_csv(self.filename)
        try:
            assert self.df.shape[1] == 3
        except AssertionError:
            print(self.df.shape)
            raise ValueError("The CSV file with the training data needs to have 3 columns: "
                             "first two with the needed input text and the third for the targets")

        self.labelizer = None
        self.cfg = AutoConfig.from_pretrained(BERT_MODEL_NAME)
        self.cfg.num_labels = self.df.overall.nunique()
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME, config=self.cfg)
        self.preprocess()

    def preprocess(self, test_ratio=0.15):
        self.df = self.df.dropna()
        self.get_labelizer()

        train_idx, test_idx= train_test_split(
            np.arange(len(self.df)),
            test_size=test_ratio,
            shuffle=True,
            stratify=self.df.iloc[:, 2],
            random_state=1
        )

        all_data = AMR(self.df, self.tokenizer, MAX_LEN)
        self.train_dataset = Subset(all_data, train_idx)
        self.val_dataset = Subset(all_data, test_idx)

    def get_labelizer(self):
        try:
            self.labelizer = load(f"{self.serialization_path}target.labelizer")
            self.df.iloc[:, 2] = self.labelizer.transform(self.df.iloc[:, 2].tolist())
        except FileNotFoundError:
            self.labelizer = LabelEncoder()
            self.df.iloc[:, 2] = self.labelizer.fit_transform(self.df.iloc[:, 2].tolist())
            dump(self.labelizer, f"{self.serialization_path}target.labelizer")

    def compute_metrics(self, p):
        predictions, true_labels = p
        true_predictions = np.argmax(predictions, axis=-1)
        results = classification_report(
            true_labels, true_predictions, output_dict=True
        )
        return {
            "precision": results["macro avg"]["precision"],
            "recall": results["macro avg"]["recall"],
            "f1": results["macro avg"]["f1-score"],
        }

    def train_model(self):
        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            num_train_epochs=1,              # total number of training epochs
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

        model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME, config=self.cfg)
        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=self.train_dataset,         # training dataset
            eval_dataset=self.val_dataset,             # evaluation dataset
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        trainer.save_model(self.serialization_path)
        self.tokenizer.save_pretrained(self.serialization_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Training a BERTlike model with the CLI. The exact model is specified in the definitions.py.')
    parser.add_argument("-fn", help="Path to the CSV file name with 3 columns: first two need to contain strings while the last contains targets to train for.", default=None)
    parser.add_argument("-sn", help="Path to the CSV file name with 3 columns: first two need to contain strings while the last contains targets to train for.", default=None)
    args = parser.parse_args()

    mt = ModelTrainer(args.fn, args.sn)
    mt.train_model()
