from transformers import (AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding,
                          Trainer, TrainingArguments)
from sklearn.metrics import accuracy_score
import numpy as np
import os
from my_dataset import get_dataset
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
warnings.filterwarnings("ignore")

checkpoint = "nghuyong/ernie-1.0-base-zh"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def model_init():
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=6)
    return model


def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy}


def main():
    uttr_dataset = get_dataset()
    tokenized_datasets = uttr_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir="ernie",
        evaluation_strategy="epoch",
        save_strategy='epoch',
        num_train_epochs=10,
        learning_rate=5e-5,
        weight_decay=0.005,
        warmup_steps=1000,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()


if __name__ == "__main__":
    main()