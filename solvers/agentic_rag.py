"""Agent 时代 · Agentic RAG —— 智能体自己决定「检索什么」，多跳推理。
和普通 RAG（一次检索）不同：这里大模型用 ReAct 循环，先检索一条、读完再决定下一步检索什么，
把分散在不同资料里的线索串起来回答。工具是 retrieve[查询]（基于嵌入相似度返回最相关的一条）。

运行：  python solvers/agentic_rag.py
输出：  rl-lab/src/data/agentic-rag.json   （agent 家族，复用 AgentPlot）
"""

import json
import math
import os
import re
import urllib.request

EMB_MODEL = "qwen3-embedding:0.6b"
GEN_MODEL = "qwen2.5:7b"

QUESTION = "鸿蒙科技的 CEO 毕业于哪所大学？"
KB = [
    "鸿蒙科技的 CEO 是张伟。",
    "张伟本科毕业于清华大学计算机系。",
    "星河传媒的 CEO 是李娜。",
    "李娜毕业于北京大学新闻学院。",
    "鸿蒙科技主要做芯片设计。",
    "星河传媒主要做影视制作。",
    "清华大学位于北京市海淀区。",
]

SYSTEM = """你是一个会检索的智能体，用 ReAct 方式回答问题。
可用工具：retrieve[查询词] —— 从知识库里检索最相关的一条资料。
资料往往分散，你可能需要检索多次、把线索串起来。严格按格式，每轮只输出一步：
Thought: <思考下一步要查什么>
Action: retrieve[<查询词>]
我会把检索到的资料作为 Observation 返回。掌握足够信息后用：
Answer: <最终答案>"""

_kb_vecs = None


def embed(texts):
    req = urllib.request.Request(
        "http://localhost:11434/api/embed",
        data=json.dumps({"model": EMB_MODEL, "input": texts}).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=120))["embeddings"]


def cos(a, b):
    d = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1
    nb = math.sqrt(sum(x * x for x in b)) or 1
    return d / (na * nb)


def retrieve(query):
    global _kb_vecs
    if _kb_vecs is None:
        _kb_vecs = embed(KB)
    qv = embed([query])[0]
    best = max(range(len(KB)), key=lambda i: cos(qv, _kb_vecs[i]))
    return KB[best]


def generate(prompt):
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(
            {"model": GEN_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.1, "num_predict": 160, "stop": ["Observation:"]}}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=180))["response"].strip()


def main():
    steps = [{"type": "task", "text": QUESTION}]
    transcript = f"{SYSTEM}\n\n问题：{QUESTION}\n"
    for _ in range(6):
        out = generate(transcript)
        th = re.search(r"Thought:\s*(.+)", out)
        if th:
            steps.append({"type": "thought", "text": th.group(1).strip()})
        ans = re.search(r"Answer:\s*(.+)", out, re.S)
        if ans:
            steps.append({"type": "answer", "text": ans.group(1).strip()})
            break
        act = re.search(r"retrieve\[(.+?)\]", out)
        if act:
            q = act.group(1)
            doc = retrieve(q)
            steps.append({"type": "action", "text": f"retrieve[{q}]"})
            steps.append({"type": "observation", "text": doc})
            transcript += f"{out}\nObservation: {doc}\n"
        else:
            transcript += out + "\n"
            if not th:
                break

    print("步骤：")
    for s in steps:
        print(" ", s["type"], "|", s["text"][:50])

    frames = [{"iter": i, "state": {"task": QUESTION, "steps": steps, "shown": i + 1}, "metrics": {"step": i}} for i in range(len(steps))]

    meta = {
        "id": "agentic-rag",
        "title": "Agentic RAG · 多跳检索",
        "family": "agent",
        "algorithm": "Agentic RAG (Qwen)",
        "description": "智能体自己决定检索什么：多跳——先查到 CEO 是谁，再查这个人毕业于哪，串起分散的线索。",
        "insight": "普通 RAG 只检索一次；Agentic RAG 让大模型像侦探一样多轮检索、边查边想，能回答需要串联多条资料的复杂问题。",
        "tutorial": {
            "problem": "答案分散在多条资料里、一次检索抓不全怎么办？让智能体自己决定「下一步查什么」，多跳检索。",
            "intuition": "普通 RAG 是「问一次、查一次、答一次」。Agentic RAG 把检索变成智能体手里的工具：它先想「要回答这个得先知道 CEO 是谁」→ 检索 → 读到「是张伟」→ 再想「那张伟毕业于哪」→ 再检索 → 串起线索给出答案。这就是 Deep Research、深度问答类 Agent 的核心。",
            "watch": [
                "蓝=思考，绿=检索(retrieve)，橙=检索到的资料，金=最终答案",
                "第一次检索拿到「CEO 是张伟」，第二次才去查「张伟毕业于哪」——多跳",
                "模型自己决定查什么、查几次，不是写死的流程",
            ],
            "concepts": [
                {"term": "多跳检索", "explain": "一次查不全，分多步检索把线索串起来"},
                {"term": "检索作为工具", "explain": "把向量检索包成 Agent 能调用的工具(retrieve)"},
                {"term": "Deep Research", "explain": "Agentic RAG 是深度研究/复杂问答 Agent 的基础范式"},
            ],
            "tryThis": "对比「RAG」(一次检索) 和这个 (多跳)：拖时间轴看它怎么先查 CEO、再查学历，两步串起答案。",
        },
        "hyperparams": {"embed": EMB_MODEL, "gen": GEN_MODEL, "tool": "retrieve", "kb": len(KB)},
    }

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "rl-lab", "src", "data", "agentic-rag.json")
    json.dump({"meta": meta, "frames": frames}, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print("写出", os.path.relpath(out, os.path.join(here, "..")), "| 帧", len(frames))


if __name__ == "__main__":
    main()
