import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

from definitions import AMR_ROOT
from definitions import BERT_ARTIFACTS
from definitions import MAX_LEN


class BertlikeAnnotator:

    def __init__(self):
        self.load_model()

    def load_model(self):
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        cfg = AutoConfig.from_pretrained(BERT_ARTIFACTS, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(BERT_ARTIFACTS,
                                                                        local_files_only=True,
                                                                        config=cfg).to(self._device)
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_ARTIFACTS,
                                                       local_files_only=True,
                                                       config=cfg)

    def get_predictions(self, data: dict) -> list:
        amr_data = self.tokenizer(
            data['summary'],
            data['review'],
            max_length=MAX_LEN,
            padding='max_length',
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True,
            return_tensors="pt")
        rs = self.model(**amr_data.to(self._device))
        sm = torch.nn.Softmax(dim=-1)
        probs = sm(rs.logits)
        target_classes = torch.argmax(probs, dim=-1).tolist()
        return target_classes


if __name__ == "__main__":
    pr = BertlikeAnnotator()
    df_test = pd.read_csv(f'{AMR_ROOT}/data/test.csv', chunksize=40)

    CHUNK_SIZE = 16

    labels = []
    ids = []
    for data in tqdm(df_test, desc="Processing test data"):
        data = data.dropna()
        target_classes = pr.get_predictions({
                'summary': data.summary.tolist(),
                'review': data.reviewText.tolist()
            })

        labels.extend(target_classes)
        ids.extend(data.id.tolist())
    df_final = pd.DataFrame({
        'id': ids,
        'predictions': labels
    })
    df_final['predictions'] = df_final['predictions'].apply(lambda x: int(x)+1)
    df_final.to_csv(f'{AMR_ROOT}/results/test_results.csv', index=False)
