from transformers import BertTokenizerFast, BertForSequenceClassification


def train_BERT():
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese')