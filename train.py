from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.optim import AdamW
from DataLoader import get_dataloader
from utils import fix_seed, get_device
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def train_epoch(model, optimizer, tr_loader, device):
    model.train()

    train_loss = 0
    for batch in tqdm(tr_loader):
        optimizer.zero_grad()
        batch = tuple(t.to(device) for t in batch)
        X, y, attention_mask = batch
        output = model(X, attention_mask=attention_mask, labels=y)

        loss = output.loss
        loss.backward()
        train_loss += loss.item()

    return train_loss / len(tr_loader)


def evaluate(model, val_loader, device):
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []

    for batch in val_loader:
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
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=6)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=0.0001)
    for epoch in range(epochs):
        train_loss = train_epoch(model, optimizer, tr_loader, device)
        val_loss, accuracy = evaluate(model, val_loader, device)
        print(
            "[{:03d}/{:03d}] Train loss: {:3.6f} | Val loss: {:3.6f} Acc: {:3.6f}".format(epoch + 1, epochs, train_loss,
                                                                                          val_loss, accuracy))


if __name__ == "__main__":
    device = get_device()
    train_loader, val_loader = get_dataloader(0.9, 16, 2)
    train_BERT(train_loader, val_loader, 10, device)
