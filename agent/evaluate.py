"""Evaluates the support agent on a fixed question set (data/eval_questions.json).

Metrics:
- Tool accuracy: did the agent call the tool we expected for the question?
- Answer check: does the final answer contain the expected facts (keywords)?
"""

from __future__ import annotations

import json

from .support_agent import SupportAgent
from .tools import DATA_DIR


def run_eval(llm, path=DATA_DIR / "eval_questions.json", verbose=False) -> dict:
    cases = json.loads(path.read_text())
    rows = []
    for case in cases:
        agent = SupportAgent(llm)  # fresh conversation for every question
        result = agent.ask(case["question"])
        used = [t["tool"] for t in result.tool_calls]
        tool_ok = all(t in used for t in case["expected_tools"])
        answer = result.answer.lower()
        answer_ok = all(k.lower() in answer for k in case["expected_keywords"])
        rows.append({"question": case["question"], "tools_used": used, "tool_ok": tool_ok,
                     "answer_ok": answer_ok, "steps": result.steps, "answer": result.answer})
        if verbose:
            mark = "PASS" if tool_ok and answer_ok else "FAIL"
            print(f"[{mark}] {case['question']}\n       tools={used}\n       answer={result.answer[:150]}")

    n = len(rows)
    return {
        "questions": n,
        "tool_accuracy": round(sum(r["tool_ok"] for r in rows) / n * 100, 1),
        "answer_accuracy": round(sum(r["answer_ok"] for r in rows) / n * 100, 1),
        "avg_steps": round(sum(r["steps"] for r in rows) / n, 2),
        "rows": rows,
    }
