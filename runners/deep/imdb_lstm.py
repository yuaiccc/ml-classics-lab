"""ML-Lab M4：IMDb LSTM 隐状态演化，逐 epoch 导出（复用 image-grid 家族）。

小 LSTM 做情感分类。每个 epoch 对固定探针评论导出每个时间步的隐状态向量
（timesteps × hidden）热图，观察隐状态随训练组织出结构。

运行：poetry run python runners/deep/imdb_lstm.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "runners" / "common"))
from trajectory import Meta, Trajectory  # noqa: E402
from capture_logger import frames_dir  # noqa: E402

OUTPUT = frames_dir() / "imdb-lstm.json"
CSV = REPO_ROOT / "data" / "kaggle" / "imdb" / "IMDB Dataset.csv"

SEED = 42
N_TRAIN = 4000
N_VAL = 1000
VOCAB_SIZE = 8000
MAX_LEN = 64
PROBE_LEN = 16
EMBED = 32
HIDDEN = 24
EPOCHS = 8
BATCH = 64
LR = 1e-3

TOKEN_RE = re.compile(r"[a-z']+")


def tokenize(t: str) -> list[str]:
    return TOKEN_RE.findall(t.lower())


def build_vocab(texts: list[str]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for t in texts:
        c.update(tokenize(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, _ in c.most_common(VOCAB_SIZE - 2):
        vocab[w] = len(vocab)
    return vocab


def encode(tokens: list[str], vocab: dict[str, int], length: int) -> list[int]:
    ids = [vocab.get(t, 1) for t in tokens[:length]]
    return ids + [0] * (length - len(ids))


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMBED, padding_idx=0)
        self.lstm = nn.LSTM(EMBED, HIDDEN, batch_first=True)
        self.head = nn.Linear(HIDDEN, 2)

    def forward(self, ids: torch.Tensor):
        out, (h, _) = self.lstm(self.emb(ids))  # out:(B,L,H) h:(1,B,H)
        return self.head(h[-1]), out


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("加载 IMDb ...")
    df = pd.read_csv(CSV).sample(N_TRAIN + N_VAL, random_state=SEED).reset_index(drop=True)
    texts = df["review"].tolist()
    labels = (df["sentiment"] == "positive").astype(int).tolist()
    tr_t, tr_y = texts[:N_TRAIN], labels[:N_TRAIN]
    va_t, va_y = texts[N_TRAIN:], labels[N_TRAIN:]

    vocab = build_vocab(tr_t)

    def to_tensor(ts: list[str], length: int) -> torch.Tensor:
        return torch.tensor([encode(tokenize(t), vocab, length) for t in ts], dtype=torch.long)

    x_train = to_tensor(tr_t, MAX_LEN)
    y_train = torch.tensor(tr_y, dtype=torch.long)
    x_val = to_tensor(va_t, MAX_LEN)
    y_val = torch.tensor(va_y, dtype=torch.long)

    probe_text = min(
        (t for t, lab in zip(va_t, va_y) if lab == 0 and len(tokenize(t)) >= PROBE_LEN),
        key=lambda t: len(tokenize(t)),
    )
    probe_tokens = tokenize(probe_text)[:PROBE_LEN]
    probe_ids = torch.tensor([[vocab.get(t, 1) for t in probe_tokens]], dtype=torch.long)
    print(f"探针 tokens: {probe_tokens}")

    model = LSTMClassifier(len(vocab))
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    def val_acc() -> float:
        model.eval()
        with torch.no_grad():
            correct = 0
            for i in range(0, len(x_val), 256):
                logits, _ = model(x_val[i : i + 256])
                correct += int((logits.argmax(1) == y_val[i : i + 256]).sum().item())
        return correct / len(x_val)

    def hidden_heatmap() -> list[list[float]]:
        """探针的 (timesteps × hidden) 隐状态，tanh∈[-1,1] 映射到 [0,1]（diverging）。"""
        model.eval()
        with torch.no_grad():
            _, out = model(probe_ids)  # (1,L,H)
        h = out[0]  # (L,H)
        return [[round(0.5 + 0.5 * float(v), 3) for v in row] for row in h]

    meta = Meta(
        id="imdb-lstm",
        title="IMDb · LSTM 隐状态",
        algorithm="LSTM",
        category="deep",
        source="python",
        abstract="循环网络逐词读入评论，隐状态是它对『目前为止读到的内容』的记忆。热图每行是一个时间步的隐状态向量，观察它随训练从杂乱逐渐组织出结构。",
        description=f"单层 LSTM（hidden={HIDDEN}）做 IMDb 情感分类，每 epoch 导出固定探针评论每个时间步的隐状态向量。",
        insight="LSTM 用门控机制选择性地记忆/遗忘，隐状态随序列推进不断更新。训练让隐状态学会编码情感相关的累积信息——这是序列建模与注意力之前的主流范式。",
        hyperparams={"lr": LR, "embed": EMBED, "hidden": HIDDEN, "max_len": MAX_LEN, "vocab_size": len(vocab)},
    )
    traj = Trajectory(meta)

    def add(epoch: int, loss: float, acc: float) -> None:
        traj.add_frame(
            iter=epoch,
            metrics={"loss": round(loss, 4), "val_accuracy": round(acc, 4)},
            family="image-grid",
            data={
                "probeTokens": probe_tokens,
                "groups": [{
                    "title": "探针隐状态 (行=时间步/词, 列=hidden 单元)",
                    "w": HIDDEN, "h": PROBE_LEN, "colormap": "diverging",
                    "images": [sum(hidden_heatmap(), [])],
                }],
            },
        )

    add(0, 0.0, val_acc())
    print("开始训练 ...")
    n = len(x_train)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i : i + BATCH]
            logits, _ = model(x_train[idx])
            loss = loss_fn(logits, y_train[idx])
            optim.zero_grad(); loss.backward(); optim.step()
            total += loss.item() * len(idx)
        tl = total / n
        acc = val_acc()
        add(epoch, tl, acc)
        print(f"epoch {epoch}: loss={tl:.4f} val_acc={acc:.4f}")

    traj.write(OUTPUT)
    print(f"已写出 {len(traj.frames)} 帧 -> {OUTPUT}")


if __name__ == "__main__":
    main()
