import torch
from data_loader import get_dataloader
from utils import fix_seed, get_device
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import argparse
from transformers import AutoTokenizer


def test_model(model, val_loader, device):
    model.eval()
    val_preds = []
    val_labels = []

    for batch in tqdm(val_loader):
        with torch.no_grad():
            batch = tuple(t.to(device) for t in batch)
            X, y, attention_mask = batch
            output = model(X, attention_mask=attention_mask, labels=y)

            logits = output.logits
            preds = logits.argmax(dim=1)
            val_preds.extend(preds.cpu().tolist())
            val_labels.extend(y.cpu().tolist())

    accuracy = accuracy_score(val_labels, val_preds)
    scores = precision_recall_fscore_support(val_labels, val_preds, average='weighted')
    report = classification_report(val_labels, val_preds)
    cm = confusion_matrix(val_labels, val_preds)

    return accuracy, scores[:3], report, cm


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert-base-chinese', type=str, help='使用什么预训练模型')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    fix_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader, val_loader = get_dataloader(0.9, 32, 2, tokenizer)

    model = torch.load('model.pth').to('cuda')
    accuracy, scores, report, conf_matrix = test_model(model, val_loader, device)

    print(report)
    print(conf_matrix)