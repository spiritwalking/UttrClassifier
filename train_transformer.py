import torch
import torch.nn as nn
from model import TransformerModel
from sklearn.metrics import accuracy_score
from utils import get_device
from data_loader import get_dataloader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BertTokenizerFast


def train_epoch(model, optimizer, tr_loader, device, criterion):
    model.train()

    train_loss = 0
    for X, y, _ in tqdm(tr_loader):
        X, y = X.to(device, dtype=torch.float32), y.to(device)
        output = model(X)

        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()


    return train_loss / len(tr_loader)


def evaluate(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []

    for X, y, _ in tqdm(val_loader):
        with torch.no_grad():
            X, y = X.to(device, dtype=torch.float32), y.to(device)
            output = model(X)
            loss = criterion(output, y)

            preds = output.argmax(dim=1)
            val_loss += loss.item()

            val_preds.extend(preds.cpu().tolist())
            val_labels.extend(y.cpu().tolist())

    accuracy = accuracy_score(val_labels, val_preds)
    return val_loss / len(val_loader), accuracy


def train_Transformer(tr_loader, val_loader, epochs, device):
    model = TransformerModel()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loss = train_epoch(model, optimizer, tr_loader, device, criterion)
        val_loss, accuracy = evaluate(model, val_loader, device, criterion)
        print(f"[{epoch + 1:03d}/{epochs:03d}] Train loss: {train_loss:3.6f} | Val loss: {val_loss:3.6f} Acc: {accuracy:3.6f}")


if __name__ == "__main__":
    # device = get_device()
    device = 'cuda:7'
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    train_loader, val_loader = get_dataloader(0.9, 64, 4, tokenizer)
    train_Transformer(train_loader, val_loader, 10, device)
