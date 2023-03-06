import json
import codecs
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizerFast


class myDataset(Dataset):
    def __init__(self):
        dialog_list = json.loads(codecs.open("data/dialog_release.json", "r", "utf-8").read())
        document_list = json.loads(codecs.open("data/document_url_release.json", "r", "utf-8").read())

        self.data = []
        for dialog in dialog_list:
            for uttr in dialog['content'][2:-2]:
                document_id = dialog['document_id']
                self.data.append([uttr, document_list[document_id]['topic']])  # [[uttr, topic], ...]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        uttr, topic = self.data[item]
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
        encoded_text = tokenizer(
            uttr,
            add_special_tokens=True,
            max_length=20,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'  # 返回PyTorch张量
        )

        topic2label = {'体育': 0, '娱乐': 1, '科技': 2, '游戏': 3, '教育': 4, '健康': 5}
        label = topic2label[topic]

        return encoded_text['input_ids'][0], label, encoded_text['attention_mask'][0]


def get_dataloader(ratio, batch_size, n_worfers):
    my_dataset = myDataset()

    # trainlen = int(ratio * len(my_dataset))
    # lengths = [trainlen, len(my_dataset) - trainlen]

    # TODO: Change lengths
    lengths = [100, 100, len(my_dataset)-200]
    trainset, validset, a = random_split(my_dataset, lengths)

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
    train_loader, valid_loader = get_dataloader(0.9, batch_size=32, n_worfers=2)
    for X, y, mask in train_loader:
        print(X)
        break
