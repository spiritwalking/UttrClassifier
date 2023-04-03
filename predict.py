from transformers import (AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding,
                          Trainer, TrainingArguments)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import os
from my_dataset import get_dataset
from utils import fix_seed
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")

checkpoint = "nghuyong/ernie-1.0-base-zh"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained("ernie1/checkpoint-6759")


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


def main():
    fix_seed(42)
    uttr_dataset = get_dataset()
    tokenized_datasets = uttr_dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir='test-ernie',
        resume_from_checkpoint="ernie1/checkpoint-6759",
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
    # trainer.evaluate()
    predictions, labels, metrics = trainer.predict(tokenized_datasets["test"])
    print(metrics['test_accuracy'])

    # topic2label = ['音乐', '电影', '旅行', '教育', '科技', '体育']
    # wrong_predictions = []
    # for i, prediction in enumerate(predictions):
    #     predicted_label = prediction.argmax()
    #     true_label = int(labels[i])
    #     if predicted_label != true_label:
    #         topic = topic2label[tokenized_datasets["test"][i]['label']]
    #         wrong_predictions.append([tokenized_datasets["test"][i]['text'], topic])
    #
    # with open("wrong.json", 'w') as f:
    #     json.dump(wrong_predictions, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
