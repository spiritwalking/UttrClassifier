import json
import codecs
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizerFast
import random


class NaturalConvDataset(Dataset):
    def __init__(self, tokenizer):
        dialog_list = json.loads(codecs.open("data/dialog_release.json", "r", "utf-8").read())
        document_list = json.loads(codecs.open("data/document_url_release.json", "r", "utf-8").read())

        self.tokenizer = tokenizer
        self.data = []
        for dialog in dialog_list:
            for uttr in dialog['content'][2:-2]:
                document_id = dialog['document_id']
                self.data.append([uttr, document_list[document_id]['topic']])  # [[uttr, topic], ...]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        uttr, topic = self.data[item]
        encoded_text = self.tokenizer(
            uttr,
            add_special_tokens=True,
            max_length=30,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'  # 返回PyTorch张量
        )

        topic2label = {'体育': 0, '娱乐': 1, '科技': 2, '游戏': 3, '教育': 4, '健康': 5}
        label = topic2label[topic]

        return encoded_text['input_ids'][0], label, encoded_text['attention_mask'][0]


class MyDataset(Dataset):
    def __init__(self, tokenizer):
        self.data = []
        with open('data/data.json', 'r', encoding='utf-8') as f:
            dialogs = json.load(f)
            for dialog in dialogs:
                topic = dialog['topic']
                if topic in ['体育', '科技', '教育']:
                    dialog['text'] = dialog['text'][2:-2]

                for uttr in dialog['text']:
                    self.data.append([uttr, topic])

        # self.data = balance_data(self.data)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        uttr, topic = self.data[item]
        encoded_text = self.tokenizer(
            uttr,
            add_special_tokens=True,
            max_length=30,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'  # 返回PyTorch张量
        )

        topic2label = {'体育': 0, '科技': 1, '教育': 2, '旅行': 3, '电影': 4, '音乐': 5}
        label = topic2label[topic]

        return encoded_text['input_ids'][0], label, encoded_text['attention_mask'][0]


def get_dataset(ratio, tokenizer):
    my_dataset = MyDataset(tokenizer)

    trainlen = int(ratio * len(my_dataset))
    lengths = [trainlen, len(my_dataset) - trainlen]

    trainset, validset = random_split(my_dataset, lengths)
    return trainset, validset


def balance_data(data):
    sport = [u for u in data if u[1] == '体育']
    not_sport = [u for u in data if u[1] != '体育']
    sport = random.sample(sport, 80000)
    not_sport.extend(sport)
    return not_sport


def get_dataloader(ratio, batch_size, n_worfers, tokenizer):
    trainset, validset = get_dataset(ratio, tokenizer)

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_worfers
    )

    valid_loader = DataLoader(
        validset,
        batch_size=batch_size,
        num_workers=n_worfers
    )

    return train_loader, valid_loader


if __name__ == "__main__":
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    train_loader, valid_loader = get_dataloader(0.9, batch_size=32, n_worfers=0, tokenizer=tokenizer)
    for X, y, mask in train_loader:
        print(X)
        break
