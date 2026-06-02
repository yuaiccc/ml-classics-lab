"""大模型时代 · 真实语义嵌入 —— 用本地 Ollama 的 qwen3-embedding 把一批词转成 1024 维向量，
算两两余弦相似度，导出给前端做 MDS 可视化（前端把高维相似度还原成 2D 语义簇）。

运行：  python solvers/embed_words.py   （需本地 Ollama 在 11434 且已拉 qwen3-embedding:0.6b）
输出：  rl-lab/src/data/qwen-embeddings.json
"""

import json
import math
import os
import urllib.request

MODEL = "qwen3-embedding:0.6b"
GROUPS = {
    "动物": ["猫", "狗", "老虎", "大象", "兔子"],
    "国家": ["中国", "日本", "法国", "美国", "巴西"],
    "食物": ["米饭", "面包", "苹果", "牛奶", "披萨"],
    "情绪": ["快乐", "悲伤", "愤怒", "恐惧", "惊讶"],
    "颜色": ["红色", "蓝色", "绿色", "黄色", "紫色"],
}


def main():
    words, group_idx, group_names = [], [], list(GROUPS.keys())
    for gi, ws in enumerate(GROUPS.values()):
        for w in ws:
            words.append(w)
            group_idx.append(gi)

    req = urllib.request.Request(
        "http://localhost:11434/api/embed",
        data=json.dumps({"model": MODEL, "input": words}).encode(),
        headers={"Content-Type": "application/json"},
    )
    emb = json.load(urllib.request.urlopen(req, timeout=120))["embeddings"]

    def norm(v):
        n = math.sqrt(sum(x * x for x in v)) or 1
        return [x / n for x in v]

    E = [norm(v) for v in emb]
    n = len(E)
    sim = [[round(sum(E[i][k] * E[j][k] for k in range(len(E[i]))), 4) for j in range(n)] for i in range(n)]

    same = sum(
        group_idx[max((j for j in range(n) if j != i), key=lambda j: sim[i][j])] == group_idx[i] for i in range(n)
    )
    print(f"嵌入 {n}×{len(E[0])} | 最近邻同组率 {same}/{n} = {same/n:.2f}")

    out = {"words": words, "groups": group_idx, "groupNames": group_names, "sim": sim}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rl-lab", "src", "data", "qwen-embeddings.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    print("写出", os.path.relpath(path))


if __name__ == "__main__":
    main()
