from datasets import Dataset
import json


def gen():
    with open('data/data.json', 'r', encoding='utf-8') as f:
        dialogs = json.load(f)
        for dialog in dialogs:
            topic = dialog['topic']
            if topic in ['体育', '科技', '教育']:
                dialog['text'] = dialog['text'][2:-2]

            for uttr in dialog['text']:
                if len(uttr) <= 12:  # 滤除信息量较少的句子
                    continue
                yield {"text": uttr, "label": topic}


def get_dataset():
    dataset = Dataset.from_generator(gen)
    # dataset = dataset.select(range(2000))
    dataset = dataset.class_encode_column("label")  # change type(label) from String to ClassLabel

    topic2label = {'音乐': 0, '电影': 1, '旅行': 2, '教育': 3, '科技': 4, '体育': 5}
    dataset_aligned = dataset.align_labels_with_mapping(topic2label, 'label')
    return dataset_aligned.train_test_split(test_size=0.05)


if __name__ == "__main__":
    uttr_dataset = get_dataset()
    print(uttr_dataset['test'])
