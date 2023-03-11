import torch
from transformers import BertForSequenceClassification, BertConfig
from torch.optim import AdamW
from DataLoader import get_dataloader
from utils import fix_seed, get_device
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim.lr_scheduler import LinearLR
from torch.nn import CrossEntropyLoss


def train_epoch(model, optimizer, tr_loader, device, criterion):
    model.train()


    train_loss = 0
    pbar = tqdm(tr_loader)
    for batch in pbar:
        optimizer.zero_grad()
        batch = tuple(t.to(device) for t in batch)
        X, y, attention_mask = batch

        output = model(X, attention_mask=attention_mask, labels=y)

        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # debug
        pbar.set_description(f"Loss: {loss.item():.6f}")

        train_loss += loss.item()

    return train_loss / len(tr_loader)


def evaluate(model, val_loader, device):
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []

    for batch in tqdm(val_loader):
        with torch.no_grad():
            batch = tuple(t.to(device) for t in batch)
            X, y, attention_mask = batch
            output = model(X, attention_mask=attention_mask, labels=y)
            loss = output.loss
            val_loss += loss.item()

            logits = output.logits
            preds = logits.argmax(dim=1)
            val_preds.extend(preds.cpu().tolist())
            val_labels.extend(y.cpu().tolist())

    accuracy = accuracy_score(val_labels, val_preds)
    scores = precision_recall_fscore_support(val_labels, val_preds, average='weighted')

    return val_loss / len(val_loader), accuracy, scores[:2]


def train_BERT(tr_loader, val_loader, epochs, device):
    config = BertConfig.from_pretrained('bert-base-chinese', num_labels=6, hidden_dropout_prob=0.3,
                                        classifier_dropout=0.3, attention_probs_dropout_prob=0.3)
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', config=config).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.015)
    scheduler = LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=10)
    criterion = CrossEntropyLoss(weight=torch.tensor([1,1,1,5,1,5],dtype=torch.float))

    best_acc = 0
    for epoch in range(epochs):
        train_loss = train_epoch(model, optimizer, tr_loader, device, criterion)
        val_loss, accuracy, scores = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"[{epoch + 1:03d}/{epochs:03d}] Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f} "
              f"Acc: {accuracy:.6f} Precision: {scores[0]:.6f} Recall: {scores[1]:.6f}")
        if accuracy > best_acc:
            torch.save(model, 'model.pth')
            print(f"Best model saved(accuracy: {accuracy})")
            best_acc = accuracy


if __name__ == "__main__":
    fix_seed(42)
    device = 'cuda'
    print(f"Using device: {device}")
    print("Start preparing data")
    train_loader, val_loader = get_dataloader(0.9, 32, 2)
    print("Start training")
    train_BERT(train_loader, val_loader, 20, device)
