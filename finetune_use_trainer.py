from transformers import (AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding,
                          Trainer, TrainingArguments)
import evaluate
import numpy as np
import os
from my_dataset import get_dataset
from utils import fix_seed
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
warnings.filterwarnings("ignore")

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=6)


def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)


def compute_metrics(eval_preds):
    metric = evaluate.load("accuracy")
    logits, labels = eval_preds
    print(f"len{len(labels)}")
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def main():
    fix_seed(42)
    uttr_dataset = get_dataset()
    tokenized_datasets = uttr_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir="test-trainer",
        evaluation_strategy="epoch",
        save_strategy='epoch',
        num_train_epochs=10,
        learning_rate=2e-5,
        weight_decay=0.001,
        warmup_steps=2000,
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()


if __name__ == "__main__":
    main()
