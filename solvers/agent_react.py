"""Agent 时代 · ReAct 循环 —— 用本地 Ollama 的 Qwen 跑一个真实的「思考→行动→观察」智能体。
配一个计算器工具，让模型分步推理、调用工具、读结果，最后给答案。
把每一步导出成前端帧（agent 家族），逐步动画回放。

运行：  python solvers/agent_react.py
输出：  rl-lab/src/data/agent-react.json
"""

import json
import re
import urllib.request

MODEL = "qwen2.5:7b"
TASK = "一个班有 32 名学生，每人交 15 元班费，已全部收齐。班级买了 3 箱饮料，每箱 89 元；又买了 1 个篮球 128 元。剩下的钱平分给 4 个活动小组，每组能分到多少元？"

SYSTEM = """你是一个会使用工具的智能体，用 ReAct 方式分步解题。
可用工具：calc[算式] —— 计算一个数学算式，例如 calc[32*15]。
严格按下面格式，每轮只输出一个 Thought 和（可选）一个 Action：
Thought: <你的思考>
Action: calc[<算式>]
我会把计算结果作为 Observation 返回给你。当你能给出最终答案时，用：
Answer: <最终答案>
现在开始解题。"""


def llm(prompt: str) -> str:
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(
            {
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 220, "stop": ["Observation:"]},
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=180))["response"].strip()


def calc(expr: str) -> str:
    expr = expr.strip().replace("×", "*").replace("÷", "/")
    if not re.fullmatch(r"[0-9+\-*/.()\s]+", expr):
        return "（非法算式）"
    try:
        return str(round(eval(expr), 4))  # noqa: S307 仅限算术字符
    except Exception as e:
        return f"（计算错误：{e}）"


def main():
    steps = [{"type": "task", "text": TASK}]
    transcript = f"{SYSTEM}\n\n问题：{TASK}\n"
    for _ in range(6):
        out = llm(transcript)
        # 解析 Thought / Action / Answer
        thought = re.search(r"Thought:\s*(.+)", out)
        if thought:
            steps.append({"type": "thought", "text": thought.group(1).strip()})
        ans = re.search(r"Answer:\s*(.+)", out, re.S)
        if ans:
            steps.append({"type": "answer", "text": ans.group(1).strip()})
            break
        action = re.search(r"calc\[(.+?)\]", out)
        if action:
            expr = action.group(1)
            obs = calc(expr)
            steps.append({"type": "action", "text": f"calc[{expr}]"})
            steps.append({"type": "observation", "text": obs})
            transcript += f"{out}\nObservation: {obs}\n"
        else:
            transcript += out + "\n"
            if not thought:
                break

    print("步骤：")
    for s in steps:
        print(" ", s["type"], "|", s["text"][:60])

    frames = []
    for i in range(len(steps)):
        frames.append({"iter": i, "state": {"task": TASK, "steps": steps, "shown": i + 1}, "metrics": {"step": i}})

    meta = {
        "id": "agent-react",
        "title": "Agent · ReAct 推理循环",
        "family": "agent",
        "algorithm": "ReAct (Qwen)",
        "envId": "ollama",
        "description": "本地 Qwen 智能体用「思考→行动→观察」循环解题：分步推理、调用计算器工具、读结果，最后作答。",
        "insight": "大模型不只是聊天——给它工具 + ReAct 框架，它就能像 agent 一样分解任务、调用工具、根据反馈继续推理。",
        "tutorial": {
            "problem": "大模型只会聊天吗？给它工具和一套「思考-行动-观察」的循环，它就变成能干活的 Agent。",
            "intuition": "ReAct = Reasoning + Acting。智能体每一步先 Thought（想下一步干什么），再 Action（调用工具，如计算器），工具返回 Observation（结果），它读完继续想下一步……如此循环直到 Answer。这就是 AutoGPT、各类 AI Agent 的基本骨架。这条轨迹是你本地 qwen2.5 真实跑出来的。",
            "watch": [
                "蓝=思考 Thought，绿=调用工具 Action，橙=工具返回 Observation，金=最终答案",
                "逐步展开：每一步都基于上一步的观察结果继续推理",
                "模型自己决定算什么、何时收尾——不是写死的流程",
            ],
            "concepts": [
                {"term": "ReAct", "explain": "推理(Reason)与行动(Act)交替：想一步、做一步、看结果、再想"},
                {"term": "工具调用", "explain": "大模型把不擅长的事（精确计算/查资料）交给外部工具"},
                {"term": "Agent", "explain": "能自主分解任务、调用工具、根据反馈迭代的大模型应用"},
            ],
            "tryThis": "拖时间轴一步步看智能体怎么把大问题拆成几次计算，最后汇总出答案。",
        },
        "hyperparams": {"model": MODEL, "tool": "calc", "framework": "ReAct"},
    }

    import os

    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, "..", "rl-lab", "src", "data", "agent-react.json")
    json.dump({"meta": meta, "frames": frames}, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
    print("写出", out_path, "| 帧数", len(frames))


if __name__ == "__main__":
    main()
