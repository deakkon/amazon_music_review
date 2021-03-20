import torch
from pandas import DataFrame
from torch.utils.data import Dataset


class AMR(Dataset):
    def __init__(self, dataframe: DataFrame, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        reviewText = self.data.reviewText.iloc[idx].strip()
        reviewSummary = self.data.summary.iloc[idx].strip()
        inputs = self.tokenizer(
            reviewSummary,
            reviewText,
            max_length=self.max_len,
            padding='max_length',
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True)
        item = {key: torch.tensor(val) for key, val in inputs.items()}
        item['labels'] = torch.tensor(self.data.overall_enc.iloc[idx],
                                          dtype=torch.long)
        return item

    def __len__(self):
        return self.len
