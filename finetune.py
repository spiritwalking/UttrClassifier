import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from torch.optim import AdamW
from data_loader import get_dataloader
from utils import fix_seed
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim.lr_scheduler import LinearLR
import argparse


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


def train_model(tr_loader, val_loader, args):
    device = args.device
    model_name = args.model_name

    config = AutoConfig.from_pretrained(model_name, num_labels=6, hidden_dropout_prob=args.drop_prob,
                                        classifier_dropout=args.drop_prob, attention_probs_dropout_prob=args.drop_prob)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=args.weight_decay)
    scheduler = LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=10)

    best_acc = 0
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, optimizer, tr_loader, device)
        val_loss, accuracy, scores = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"[{epoch + 1:03d}/{args.epochs:03d}] Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f} "
              f"Acc: {accuracy:.6f} Precision: {scores[0]:.6f} Recall: {scores[1]:.6f}")
        if accuracy > best_acc:
            torch.save(model, model_name.split('/')[-1] + '_model.pth')
            print(f"Best model saved(accuracy: {accuracy})")
            best_acc = accuracy


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', type=str, help='使用哪些显卡')
    parser.add_argument('--model_name', default='bert-base-chinese', type=str, help='使用什么预训练模型')
    parser.add_argument('--epochs', default=10, type=int, help='训练的epoch数目')
    parser.add_argument('--batch_size', default=32, type=int, help='训练的batch size')
    parser.add_argument('--weight_decay', default=0.015, type=float, help='正则项')
    parser.add_argument('--drop_prob', default=0.3, type=float, help='dropout的概率')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    fix_seed(42)
    print(f"Using device: {args.device} and pre-trained model: {args.model_name}")

    print("Start preparing data")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader, val_loader = get_dataloader(0.9, args.batch_size, 2, tokenizer)

    print("Start training")
    train_model(train_loader, val_loader, args)
