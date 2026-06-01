"""M4 · Self-Attention 可视化（纯 PyTorch）—— 大模型时代的核心机制。

玩具任务：序列反转。输入 [A,B,C,D,E,F] -> 目标 [F,E,D,C,B,A]。
一个单头自注意力层必须学会“每个输出位置去关注它的镜像位置”，
于是注意力矩阵会从噪声逐渐收敛成一条清晰的反对角线。

这就是 BERT / GPT 的发动机：Q·K^T -> softmax -> 加权 V，注意力 = 学习“看哪里”。
手写注意力（不用 nn.MultiheadAttention）以便提取权重、便于阅读。

运行：  conda run -n tianshou python solvers/train_transformer.py
输出：  rl-lab/src/data/frames/attention-reverse.json
"""

import torch
import torch.nn.functional as F
from torch import nn

from frames_io import write_trajectory

VOCAB = 6   # 用 6 个不同 token：A B C D E F
LEN = 6     # 序列长度
DIM = 24    # 嵌入维度
SEED = 0


class TinyAttention(nn.Module):
    def __init__(self, vocab: int, length: int, dim: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(length, dim)
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, vocab)
        self.dim = dim

    def forward(self, x: torch.Tensor):
        # x: [B, L]
        _, L = x.shape
        pos = torch.arange(L, device=x.device)
        h = self.tok(x) + self.pos(pos)[None]           # [B, L, D]
        q, k, v = self.Wq(h), self.Wk(h), self.Wv(h)
        scores = q @ k.transpose(1, 2) / (self.dim ** 0.5)  # [B, L, L]
        attn = F.softmax(scores, dim=-1)                # 注意力权重
        ctx = attn @ v                                  # 加权求和
        return self.out(ctx), attn                      # logits [B,L,V], attn [B,L,L]


def make_batch(n: int, g: torch.Generator):
    x = torch.randint(0, VOCAB, (n, LEN), generator=g)
    return x, x.flip(1)  # 目标 = 反转


def main():
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED)
    Xtr, Ytr = make_batch(512, g)

    model = TinyAttention(VOCAB, LEN, DIM)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    # 固定一个全不同 token 的样本，反对角线最直观
    example = torch.tensor([[0, 1, 2, 3, 4, 5]])
    example_target = example.flip(1)[0].tolist()

    epochs = 120
    frames = []
    for e in range(epochs + 1):
        with torch.no_grad():
            logits, _ = model(Xtr)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), Ytr.reshape(-1)).item()
            acc = (logits.argmax(-1) == Ytr).float().mean().item()
            ex_logits, ex_attn = model(example)
            pred = ex_logits.argmax(-1)[0].tolist()
            attn = ex_attn[0]  # [L, L]
        frames.append(
            {
                "iter": e,
                "state": {
                    "tokens": example[0].tolist(),
                    "target": example_target,
                    "pred": pred,
                    "attention": [[round(float(v), 4) for v in row] for row in attn],
                },
                "metrics": {"loss": round(loss, 4), "accuracy": round(acc, 4)},
            }
        )
        if e == epochs:
            break
        opt.zero_grad()
        out, _ = model(Xtr)
        F.cross_entropy(out.reshape(-1, VOCAB), Ytr.reshape(-1)).backward()
        opt.step()

    meta = {
        "id": "attention-reverse",
        "title": "Self-Attention · 序列反转",
        "family": "attention",
        "algorithm": "Transformer (Self-Attention)",
        "description": "单头自注意力学习把序列反转。注意力矩阵从噪声逐渐收敛成反对角线——这就是 BERT/GPT 的核心。",
        "insight": "注意力让每个位置自己决定“去看哪些位置”。反转任务里，输出位置 i 必须关注输入位置 L-1-i，于是注意力矩阵收敛成反对角线。",
        "tutorial": {
            "problem": "把一句话顺序反转。模型不能靠固定连线，必须自己学会“每个位置该关注哪里”。",
            "intuition": "自注意力 = 每个 token 发出一个“查询(Q)”，和所有 token 的“键(K)”比对，算出关注权重(softmax)，再按权重把“值(V)”加权汇总。反转任务里，模型会学到：第 i 个输出强烈关注第 L-1-i 个输入。这套 Q·K→softmax→加权V 的机制，正是 BERT / GPT / 所有大模型的发动机。",
            "watch": [
                "注意力热图：行=输出位置，列=输入位置，越亮=关注越强",
                "训练初期一片模糊（不知道看哪），逐渐收敛成一条清晰的反对角线",
                "反对角线 = “输出第1个 ← 关注输入最后一个”，正好实现反转",
                "下方预测序列从乱码逐渐变成正确的反转结果",
            ],
            "concepts": [
                {"term": "Query / Key / Value", "explain": "查询去和键比对算权重，再用权重汇总值——注意力的三件套"},
                {"term": "注意力权重", "explain": "softmax 后的矩阵，每行加起来=1，表示一个位置把注意力分给谁"},
                {"term": "自注意力", "explain": "序列内部各位置互相关注，是 Transformer / 大模型的核心层"},
            ],
            "tryThis": "拖时间轴看注意力矩阵从模糊噪声收敛成反对角线，同时下方预测从乱码变成正确反转。",
        },
        "hyperparams": {"vocab": VOCAB, "seq_len": LEN, "dim": DIM, "lr": 3e-3, "epochs": epochs},
    }
    write_trajectory("attention-reverse", meta, frames)
    print(f"Transformer 训练完成，最终 token 准确率 {frames[-1]['metrics']['accuracy']:.3f}")


if __name__ == "__main__":
    main()
