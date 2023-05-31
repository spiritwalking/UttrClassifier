from transformers import (AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding,
                          Trainer, TrainingArguments, pipeline)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import os
import json
from my_dataset import get_dataset
from trainer_utils import fix_seed
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
fix_seed(42)  # 先设置可见GPU，再固定随机种子
warnings.filterwarnings("ignore")

checkpoint = "ernie/checkpoint-14553"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions)
    cm = confusion_matrix(labels, predictions)
    print(report)
    print(cm)
    return {'accuracy': accuracy}


def classifier_pipeline(text):
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return classifier(text)


def model_evaluate(trainer, val_set):
    # trainer.evaluate()
    predictions, labels, metrics = trainer.predict(val_set)
    print(metrics['test_accuracy'])

    topic2label = ['音乐', '电影', '旅行', '科技', '体育']
    wrong_predictions = []
    for i, prediction in enumerate(predictions):
        predicted_label = prediction.argmax()
        true_label = int(labels[i])
        if predicted_label != true_label:
            topic = topic2label[val_set[i]['label']]
            wrong_predictions.append([val_set[i]['text'], topic])

    with open("wrong.json", 'w') as f:
        json.dump(wrong_predictions, f, ensure_ascii=False, indent=2)


def main():
    uttr_dataset = get_dataset()
    tokenized_datasets = uttr_dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir='test-ernie',
        resume_from_checkpoint=checkpoint,
        per_device_eval_batch_size=32
    )

    trainer = Trainer(
        model,
        args=training_args,
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    model_evaluate(trainer, tokenized_datasets["test"])


if __name__ == "__main__":
    # main()
    while True:
        request = input("> ")
        predict = classifier_pipeline(request)
        print(f"> {predict}")
