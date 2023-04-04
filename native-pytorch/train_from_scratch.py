import torch
from transformers import BertTokenizerFast, BertConfig, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from data_loader import get_dataloader
from utils import fix_seed
from tqdm import tqdm
from accelerate import Accelerator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings

warnings.filterwarnings('ignore')


def train(model, train_loader, val_loader):
    accelerator = Accelerator()
    epochs = 20
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=3000,
        num_training_steps=(len(train_loader) * epochs)
    )

    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler)

    for epoch in range(epochs):
        # train model
        model.train()
        train_loss = 0
        for X, y, attention_mask in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(X, attention_mask=attention_mask, labels=y)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()

            train_loss += loss.item()
        train_loss /= len(train_loader)

        # evaluate model
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []

        for X, y, attention_mask in val_loader:
            with torch.no_grad():
                outputs = model(X, attention_mask=attention_mask, labels=y)
                val_loss += outputs.loss.item()
                preds = outputs.logits.argmax(dim=1)
                preds, labels = accelerator.gather_for_metrics((preds, y))

                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(labels.cpu().tolist())

        val_loss /= len(val_loader)
        accuracy = accuracy_score(val_labels, val_preds)
        scores = precision_recall_fscore_support(val_labels, val_preds, average='weighted')
        accelerator.print(f"[{epoch + 1:03d}/{epochs:03d}] Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f} "
                          f"Acc: {accuracy:.6f} Precision: {scores[0]:.6f} Recall: {scores[1]:.6f}")


if __name__ == "__main__":
    fix_seed(42)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    kwargs = {'num_hidden_layers': 24, 'num_attention_heads': 16,
              'intermediate_size': 4096, 'hidden_size': 1024, 'num_labels': 6}
    config = BertConfig(**kwargs)
    model = BertForSequenceClassification(config=config)

    train_loader, val_loader = get_dataloader(0.9, 32, 2, tokenizer)

    train(model, train_loader, val_loader)
