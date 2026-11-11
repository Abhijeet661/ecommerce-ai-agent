"""Customer support agent: a tool-calling loop (reason -> call tool -> observe -> answer)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .tools import TOOL_SCHEMAS, run_tool

SYSTEM_PROMPT = """You are a polite customer support agent for an online home decor and furniture store.

Rules:
- Always use the tools to look up products, stock and orders. Never guess prices, stock, dates or tracking numbers.
- If the customer gives a product name instead of a SKU, call search_products first.
- If a customer reports a problem with an order (damaged, missing, wrong item, refund), call create_support_ticket and share the ticket ID.
- If a tool returns an error or nothing useful, say so honestly and ask for the missing detail (for example the order ID).
- Keep answers short, friendly and clear (2-4 sentences).
"""


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)  # trace of every tool the agent used
    steps: int = 0


class SupportAgent:
    def __init__(self, llm, max_steps: int = 6, verbose: bool = False):
        self.llm = llm
        self.max_steps = max_steps
        self.verbose = verbose
        self.history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def reset(self):
        self.history = self.history[:1]

    def ask(self, user_message: str) -> AgentResult:
        self.history.append({"role": "user", "content": user_message})
        trace = []

        for step in range(1, self.max_steps + 1):
            reply = self.llm.chat(self.history, tools=TOOL_SCHEMAS)

            if not reply.tool_calls:  # model is ready to answer
                answer = (reply.content or "").strip() or "Sorry, I could not find an answer."
                self.history.append({"role": "assistant", "content": answer})
                return AgentResult(answer=answer, tool_calls=trace, steps=step)

            # Record the model's tool request, then run each tool and send back the result
            self.history.append(
                {
                    "role": "assistant",
                    "content": reply.content,
                    "tool_calls": [
                        {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                        for tc in reply.tool_calls
                    ],
                }
            )
            for tc in reply.tool_calls:
                try:
                    args = json.loads(tc.arguments) if tc.arguments else {}
                except json.JSONDecodeError:
                    args, result = {}, {"error": "Arguments were not valid JSON"}
                else:
                    result = run_tool(tc.name, args)

                trace.append({"tool": tc.name, "args": args, "result": result})
                if self.verbose:
                    print(f"  [tool] {tc.name}({args}) -> {json.dumps(result)[:160]}")
                self.history.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})

        answer = "Sorry, this is taking too many steps. A human agent will follow up with you."
        self.history.append({"role": "assistant", "content": answer})
        return AgentResult(answer=answer, tool_calls=trace, steps=self.max_steps)
