import os
import re
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


MAX_VOCAB = 30000
MAX_LEN = 300
BATCH_TRAIN = 64
BATCH_VAL = 128
EMB_DIM = 128
HID = 128
LR = 2e-3
MAX_EPOCHS = 50
PATIENCE = 3
RANDOM_STATE = 42

DATA_CSV = "outputs/dataset.csv"
OUT_DIR = "outputs"
METRICS_PATH = os.path.join(OUT_DIR, "metrics_neural.json")
CM_PATH = os.path.join(OUT_DIR, "confusion_neural.csv")



df = pd.read_csv(DATA_CSV)
texts = df["text"].astype(str).tolist()
labels = df["label"].astype(str).tolist()


le = LabelEncoder()
y = le.fit_transform(labels)

X_train, X_val, y_train, y_val = train_test_split(
    texts, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

def simple_tokenize(s: str):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return s.split()


from collections import Counter
counter = Counter()
for t in X_train:
    counter.update(simple_tokenize(t))

vocab = {"<pad>": 0, "<unk>": 1}
for w, _ in counter.most_common(MAX_VOCAB - len(vocab)):
    vocab[w] = len(vocab)

def encode(text, max_len=MAX_LEN):
    toks = simple_tokenize(text)
    ids = [vocab.get(w, vocab["<unk>"]) for w in toks[:max_len]]
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    return np.array(ids, dtype=np.int64)

class TextDS(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(encode(self.X[idx])), torch.tensor(self.y[idx])

train_ds = TextDS(X_train, y_train)
val_ds = TextDS(X_val, y_val)
train_dl = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_VAL, shuffle=False)


class BiGRUClassifier(nn.Module):
    def __init__(self, vocab_size, n_classes, emb_dim=EMB_DIM, hid=HID):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hid, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hid * 2, n_classes)

    def forward(self, x):
        e = self.emb(x)                 # (B, T, emb)
        out, h = self.rnn(e)            # h: (2, B, hid)
        h = torch.cat([h[0], h[1]], dim=1)  # (B, 2*hid)
        h = self.dropout(h)
        return self.fc(h)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BiGRUClassifier(len(vocab), len(le.classes_)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
crit = nn.CrossEntropyLoss()



def evaluate(return_details=False):
    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1)
            all_p.extend(pred.cpu().numpy())
            all_y.extend(yb.cpu().numpy())

    acc = accuracy_score(all_y, all_p)
    macro_f1 = f1_score(all_y, all_p, average="macro")
    weighted_f1 = f1_score(all_y, all_p, average="weighted")

    if return_details:
        rep = classification_report(
            all_y, all_p,
            target_names=le.classes_,
            zero_division=0,
            output_dict=True
        )
        cm = confusion_matrix(all_y, all_p)
        return acc, macro_f1, weighted_f1, rep, cm, all_y, all_p

    return acc, macro_f1, weighted_f1



os.makedirs(OUT_DIR, exist_ok=True)

best_acc = -1.0
best_state = None
best_epoch = 0
no_improve = 0
history = []

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = crit(logits, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item()

    acc, macro_f1, weighted_f1 = evaluate()
    avg_loss = total_loss / max(1, len(train_dl))

    history.append({
        "epoch": epoch,
        "train_loss": avg_loss,
        "val_acc": acc,
        "val_macro_f1": macro_f1,
        "val_weighted_f1": weighted_f1
    })

    print(f"Epoch {epoch:02d} | loss={avg_loss:.4f} | val_acc={acc:.4f} | macro_f1={macro_f1:.4f} | weighted_f1={weighted_f1:.4f}")


    if acc > best_acc + 1e-6:
        best_acc = acc
        best_epoch = epoch
        no_improve = 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"Early stopping: no improvement in {PATIENCE} epochs (best_acc={best_acc:.4f} at epoch {best_epoch})")
            break


if best_state is not None:
    model.load_state_dict(best_state)

acc, macro_f1, weighted_f1, rep, cm, all_y, all_p = evaluate(return_details=True)


pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_csv(CM_PATH, encoding="utf-8")


with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump({
        "best_epoch": best_epoch,
        "best_val_acc_tracked": best_acc,
        "final_val_acc": acc,
        "final_macro_f1": macro_f1,
        "final_weighted_f1": weighted_f1,
        "per_class_report": rep,
        "history": history,
        "labels": list(le.classes_)
    }, f, indent=2)


print("\nSaved:")
print(" - Metrics :", METRICS_PATH)
print(" - ConfMat :", CM_PATH)
