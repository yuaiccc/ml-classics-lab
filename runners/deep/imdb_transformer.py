"""ML-Lab M4 竖切片：IMDb 小 Transformer 注意力热图，逐 epoch 导出。

在 IMDb 情感数据上训练一个极小的 Transformer 编码器（手写多头自注意力以便
逐头取出注意力权重）。每个 epoch 跑一条固定探针评论，抓出所有层 × 所有头的
注意力矩阵，连同 loss/val_accuracy 记成一帧，导出符合前端契约的
ml-lab/public/frames/imdb-transformer.json。

帧契约：state.family = "attention"
        state.data    = { tokens: [str], attention: [layer][head][L][L] }

运行：poetry run python runners/deep/imdb_transformer.py
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

OUTPUT = frames_dir() / "imdb-transformer.json"
CSV = REPO_ROOT / "data" / "kaggle" / "imdb" / "IMDB Dataset.csv"

# ==================== 超参数 ====================
SEED = 42
N_TRAIN = 4000
N_VAL = 1000
VOCAB_SIZE = 8000
MAX_LEN = 64          # 训练序列长度
PROBE_LEN = 16        # 探针热图尺寸（L×L，越小越可读）
D_MODEL = 64
N_LAYERS = 2
N_HEADS = 2
D_FF = 128
EPOCHS = 8
BATCH = 64
LR = 3e-4

TOKEN_RE = re.compile(r"[a-z']+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_vocab(texts: list[str]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in counter.most_common(VOCAB_SIZE - 2):
        vocab[word] = len(vocab)
    return vocab


def encode(tokens: list[str], vocab: dict[str, int], length: int) -> list[int]:
    ids = [vocab.get(t, 1) for t in tokens[:length]]
    ids += [0] * (length - len(ids))
    return ids


class MultiHeadSelfAttention(nn.Module):
    """手写多头自注意力，forward 额外返回每个头的注意力权重 (B, heads, L, L)。"""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        b, length, d = x.shape
        qkv = self.qkv(x).reshape(b, length, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B, heads, L, d_head)
        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)  # (B, heads, L, L)
        if mask is not None:
            scores = scores.masked_fill(mask[:, None, None, :] == 0, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = attn @ v  # (B, heads, L, d_head)
        out = out.transpose(1, 2).reshape(b, length, d)
        return self.out(out), attn


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        a, attn = self.attn(self.norm1(x), mask)
        x = x + a
        x = x + self.ff(self.norm2(x))
        return x, attn


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab_size, D_MODEL, padding_idx=0)
        self.pos = nn.Embedding(MAX_LEN, D_MODEL)
        self.layers = nn.ModuleList(
            [EncoderLayer(D_MODEL, N_HEADS, D_FF) for _ in range(N_LAYERS)]
        )
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, 2)

    def forward(self, ids: torch.Tensor, collect: bool = False):
        mask = (ids != 0).float()  # (B, L)
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = self.tok(ids) + self.pos(pos)[None]
        attns = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            if collect:
                attns.append(attn)
        x = self.norm(x)
        # mean-pool over非 pad token
        m = mask[:, :, None]
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1e-6)
        logits = self.head(pooled)
        return logits, attns


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("加载 IMDb 数据 ...")
    df = pd.read_csv(CSV).sample(N_TRAIN + N_VAL, random_state=SEED).reset_index(drop=True)
    texts = df["review"].tolist()
    labels = (df["sentiment"] == "positive").astype(int).tolist()

    train_texts, train_labels = texts[:N_TRAIN], labels[:N_TRAIN]
    val_texts, val_labels = texts[N_TRAIN:], labels[N_TRAIN:]

    vocab = build_vocab(train_texts)
    print(f"词表大小: {len(vocab)}")

    def to_tensor(ts: list[str], length: int) -> torch.Tensor:
        return torch.tensor([encode(tokenize(t), vocab, length) for t in ts], dtype=torch.long)

    x_train = to_tensor(train_texts, MAX_LEN)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    x_val = to_tensor(val_texts, MAX_LEN)
    y_val = torch.tensor(val_labels, dtype=torch.long)

    # 固定探针：取一条较短的负面评论，截断到 PROBE_LEN 个真实 token（无 padding）
    probe_text = min(
        (t for t, lab in zip(val_texts, val_labels) if lab == 0 and len(tokenize(t)) >= PROBE_LEN),
        key=lambda t: len(tokenize(t)),
    )
    probe_tokens = tokenize(probe_text)[:PROBE_LEN]
    probe_ids = torch.tensor([[vocab.get(t, 1) for t in probe_tokens]], dtype=torch.long)
    print(f"探针 tokens: {probe_tokens}")

    model = TinyTransformer(len(vocab))
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    def evaluate() -> float:
        model.eval()
        with torch.no_grad():
            correct = 0
            for i in range(0, len(x_val), 256):
                logits, _ = model(x_val[i : i + 256])
                correct += (logits.argmax(1) == y_val[i : i + 256]).sum().item()
        return correct / len(x_val)

    def probe_attention() -> list[list[list[list[float]]]]:
        """返回 attention[layer][head][L][L]（已四舍五入）。"""
        model.eval()
        with torch.no_grad():
            _, attns = model(probe_ids, collect=True)
        out = []
        for attn in attns:  # attn: (1, heads, L, L)
            a = attn[0]  # (heads, L, L)
            out.append([[[round(float(v), 4) for v in row] for row in head] for head in a])
        return out

    meta = Meta(
        id="imdb-transformer",
        title="IMDb · Tiny Transformer 注意力",
        algorithm="Tiny Transformer",
        category="deep",
        source="python",
        abstract="小型 Transformer 在 IMDb 情感分类上的自注意力随训练演化：观察模型如何逐渐学会聚焦情感词。帧内可切换层与注意力头。",
        description=f"{N_LAYERS} 层 × {N_HEADS} 头的极小 Transformer（d_model={D_MODEL}）做情感二分类。每个 epoch 抓取固定探针评论的全部注意力矩阵。",
        insight="自注意力让每个 token 直接看到序列中其他 token；训练中注意力权重从弥散逐渐聚焦到情感相关词上。深层与浅层、不同头关注的模式各不相同。",
        hyperparams={
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "max_len": MAX_LEN,
            "vocab_size": len(vocab),
            "lr": LR,
        },
    )
    traj = Trajectory(meta)

    print("开始训练 ...")
    n = len(x_train)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i : i + BATCH]
            logits, _ = model(x_train[idx])
            loss = loss_fn(logits, y_train[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() * len(idx)
        train_loss = total_loss / n
        val_acc = evaluate()
        attention = probe_attention()
        traj.add_frame(
            iter=epoch,
            metrics={"loss": round(train_loss, 4), "val_accuracy": round(val_acc, 4)},
            family="attention",
            data={"tokens": probe_tokens, "attention": attention},
        )
        print(f"epoch {epoch}: loss={train_loss:.4f} val_acc={val_acc:.4f}")

    traj.write(OUTPUT)
    print(f"已写出 {len(traj.frames)} 帧 -> {OUTPUT}")


if __name__ == "__main__":
    main()
