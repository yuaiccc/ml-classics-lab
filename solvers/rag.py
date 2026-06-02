"""大模型时代 · RAG（检索增强生成）—— 用本地 Ollama 的 Qwen。
流程：把问题和知识库都做嵌入 → 按相似度检索最相关的几条 → 喂给 Qwen 生成有依据的回答。
导出成帧：先看问题，再逐条揭示检索排序，最后给出生成答案。

运行：  python solvers/rag.py   （需本地 Ollama，已拉 qwen3-embedding:0.6b 和 qwen2.5:7b）
输出：  rl-lab/src/data/rag.json
"""

import json
import math
import os
import urllib.request

EMB_MODEL = "qwen3-embedding:0.6b"
GEN_MODEL = "qwen2.5:7b"
TOP_K = 3

QUERY = "什么动物跑得最快？它能跑多快？"
DOCS = [
    "猎豹是陆地上奔跑最快的动物，短距离冲刺时速可达约 120 公里。",
    "蓝鲸是地球上现存最大的动物，体长可超过 30 米。",
    "大象是陆地上最大的哺乳动物，记忆力很强、寿命很长。",
    "蜂鸟是唯一能向后飞行的鸟，每秒振翅约 50 次。",
    "树懒动作极慢，大部分时间挂在树上睡觉。",
    "章鱼有三个心脏，血液是蓝色的。",
    "帝企鹅能潜到水下 500 米深处觅食。",
    "长颈鹿是世界上最高的陆地动物，脖子特别长。",
    "蝙蝠是唯一真正会飞的哺乳动物，靠回声定位捕食。",
    "游隼俯冲时速可超过 300 公里，是飞行最快的鸟。",
]


def embed(texts):
    req = urllib.request.Request(
        "http://localhost:11434/api/embed",
        data=json.dumps({"model": EMB_MODEL, "input": texts}).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=120))["embeddings"]


def generate(prompt):
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(
            {"model": GEN_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.2, "num_predict": 160}}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=180))["response"].strip()


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb or 1)


def main():
    vecs = embed([QUERY] + DOCS)
    qv, dvs = vecs[0], vecs[1:]
    scored = sorted(
        ({"text": DOCS[i], "score": round(cos(qv, dvs[i]), 4)} for i in range(len(DOCS))),
        key=lambda d: -d["score"],
    )
    top = scored[:TOP_K]
    print("检索排序：")
    for d in scored:
        print(f"  {d['score']:.3f}  {d['text'][:30]}")

    context = "\n".join(f"- {d['text']}" for d in top)
    prompt = f"只根据下面的资料回答问题，不要编造。\n资料：\n{context}\n\n问题：{QUERY}\n回答："
    answer = generate(prompt)
    print("\n生成答案：", answer)

    n = len(scored)
    frames = []
    # 帧 0：只有问题；帧 1..n：逐条揭示检索结果；末帧：给出答案
    for shown in range(n + 1):
        frames.append(
            {
                "iter": shown,
                "state": {"query": QUERY, "docs": scored, "shown": shown, "topK": TOP_K, "answer": ""},
                "metrics": {"topScore": scored[0]["score"]},
            }
        )
    frames.append(
        {
            "iter": n + 1,
            "state": {"query": QUERY, "docs": scored, "shown": n, "topK": TOP_K, "answer": answer},
            "metrics": {"topScore": scored[0]["score"]},
        }
    )

    meta = {
        "id": "rag",
        "title": "RAG · 检索增强生成",
        "family": "rag",
        "algorithm": "RAG (Qwen)",
        "description": "本地 Qwen：把问题和知识库都做嵌入，检索最相关的几条资料，再据此生成有依据的回答。",
        "insight": "大模型会一本正经地胡说（幻觉）。RAG 给它外挂一个知识库：先检索相关资料，再让模型基于资料回答，更准、可溯源。",
        "tutorial": {
            "problem": "大模型记不住所有知识、还会一本正经地编（幻觉）。怎么让它基于「可靠资料」回答？答案是 RAG。",
            "intuition": "RAG = 检索(Retrieval) + 生成(Generation)。先把问题和知识库的每条资料都变成向量（嵌入），按相似度找出最相关的几条，把它们连同问题一起喂给大模型，让它「开卷答题」。这样回答有依据、可溯源，是企业知识库、AI 搜索、文档问答的主流方案。",
            "watch": [
                "上方是问题；中间按相似度从高到低逐条揭示知识库资料，绿色高亮的是被选中的 Top-K",
                "相似度条越长=和问题越相关；猎豹/游隼那两条分数最高",
                "最后大模型只根据选中的资料生成回答——而不是凭记忆瞎编",
            ],
            "concepts": [
                {"term": "嵌入检索", "explain": "把问题和资料都变向量，用余弦相似度找最相关的"},
                {"term": "Top-K 召回", "explain": "只取相似度最高的 K 条作为上下文，控制噪声和长度"},
                {"term": "接地生成 / 防幻觉", "explain": "让模型基于检索到的资料回答，可溯源、更可靠"},
            ],
            "tryThis": "拖时间轴看检索如何排序、Top-3 被选中，最后基于资料生成答案。这就是 AI 知识库问答的核心。",
        },
        "hyperparams": {"embed": EMB_MODEL, "gen": GEN_MODEL, "topK": TOP_K, "docs": len(DOCS)},
    }

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "rl-lab", "src", "data", "rag.json")
    json.dump({"meta": meta, "frames": frames}, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print("\n写出", os.path.relpath(out, os.path.join(here, "..")), "| 帧", len(frames))


if __name__ == "__main__":
    main()
