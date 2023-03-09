import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.optim import AdamW
from DataLoader import get_dataloader
from utils import fix_seed, get_device
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def train_epoch(model, optimizer, tr_loader, device):
    model.train()

    train_loss = 0
    pbar = tqdm(tr_loader)
    for batch in pbar:
        optimizer.zero_grad()
        batch = tuple(t.to(device) for t in batch)
        X, y, attention_mask = batch

        output = model(X, attention_mask=attention_mask, labels=y)

        loss = output.loss
        loss.backward()
        optimizer.step()

        # debug
        pbar.set_description(f"Loss: {loss.item()}")
        loss_record["train"].append(loss.item())
        # print(loss.item())

        train_loss += loss.item()

    return train_loss / len(tr_loader)


def evaluate(model, val_loader, device):
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []

    for batch in val_loader:
        with torch.no_grad():
            batch = tuple(t.to(device) for t in batch)
            X, y, attention_mask = batch
            output = model(X, attention_mask=attention_mask, labels=y)
            loss = output.loss
            logits = output.logits
            preds = logits.argmax(dim=1)

            val_loss += loss.item()

            val_preds.extend(preds.cpu().tolist())
            val_labels.extend(y.cpu().tolist())

    accuracy = accuracy_score(val_labels, val_preds)
    return val_loss / len(val_loader), accuracy


def train_BERT(tr_loader, val_loader, epochs, device):
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=6).to(device)
    optimizer = AdamW(model.parameters(), lr=0.0001)
    for epoch in range(epochs):
        train_loss = train_epoch(model, optimizer, tr_loader, device)
        val_loss, accuracy = evaluate(model, val_loader, device)
        print(f"[{epoch + 1:03d}/{epochs:03d}] Train loss: {train_loss:3.6f} | Val loss: {val_loss:3.6f} Acc: {accuracy:3.6f}")

    torch.save(model, 'model.pth')


if __name__ == "__main__":
    loss_record = {'train': [], 'val': []}
    device = get_device()
    print(f"Using device: {device}")
    print("Start preparing data")
    train_loader, val_loader = get_dataloader(0.9, 32, 4)
    print("Start training")
    train_BERT(train_loader, val_loader, 10, device)
