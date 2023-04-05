from transformers import (AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding,
                          Trainer, TrainingArguments)
from sklearn.metrics import accuracy_score
import numpy as np
import os
from my_dataset import get_dataset
from trainer_utils import fix_seed
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
fix_seed(42)
warnings.filterwarnings("ignore")

checkpoint = "nghuyong/ernie-1.0-base-zh"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=5, hidden_dropout_prob=0.2,
                                                           classifier_dropout=0.1)


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
        warmup_steps=500,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32
    )

    trainer = Trainer(
        model,
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
