"""Command-line entry point.

Examples:
    python main.py listings --marketplace amazon --limit 3
    python main.py ask "Where is my order ORD-5002?" --verbose
    python main.py chat
    python main.py eval
"""

from __future__ import annotations

import argparse
import csv
import json

from dotenv import load_dotenv

from agent.evaluate import run_eval
from agent.listing_agent import generate_listing
from agent.llm import LLMClient
from agent.support_agent import SupportAgent
from agent.tools import OUTPUT_DIR, PRODUCTS


def cmd_listings(llm, args):
    products = PRODUCTS[: args.limit] if args.limit else PRODUCTS
    results = []
    for p in products:
        r = generate_listing(llm, p, marketplace=args.marketplace)
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {p['sku']} {p['product_name']} (attempts: {r.attempts})")
        for issue in r.issues:
            print(f"    - {issue}")
        results.append(r)

    OUTPUT_DIR.mkdir(exist_ok=True)
    json_path = OUTPUT_DIR / f"listings_{args.marketplace}.json"
    json_path.write_text(json.dumps([r.__dict__ for r in results], indent=2))

    csv_path = OUTPUT_DIR / f"listings_{args.marketplace}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sku", "passed", "attempts", "title", "bullets", "description", "search_terms"])
        for r in results:
            L = r.listing
            writer.writerow([r.sku, r.passed, r.attempts, L.get("title", ""), " | ".join(L.get("bullets", [])),
                             L.get("description", ""), L.get("search_terms", "")])

    n = len(results)
    first_try = sum(1 for r in results if r.passed and r.attempts == 1)
    passed = sum(1 for r in results if r.passed)
    print(f"\nSummary: {n} products | passed on first try: {first_try}/{n} | "
          f"passed after self-correction: {passed}/{n}")
    print(f"Saved: {json_path} and {csv_path}")


def cmd_ask(llm, args):
    agent = SupportAgent(llm, verbose=args.verbose)
    print(agent.ask(args.question).answer)


def cmd_chat(llm, args):
    agent = SupportAgent(llm, verbose=args.verbose)
    print("Customer support agent. Type 'exit' to quit, 'reset' to start over.\n")
    while True:
        try:
            msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg.lower() in ("exit", "quit"):
            break
        if msg.lower() == "reset":
            agent.reset()
            continue
        if msg:
            print(f"Agent: {agent.ask(msg).answer}\n")


def cmd_eval(llm, args):
    report = run_eval(llm, verbose=args.verbose)
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "eval_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nQuestions: {report['questions']} | Tool accuracy: {report['tool_accuracy']}% | "
          f"Answer accuracy: {report['answer_accuracy']}% | Avg steps: {report['avg_steps']}")
    print(f"Saved: {OUTPUT_DIR / 'eval_report.json'}")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="E-commerce AI agent: listings + customer support")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("listings", help="Generate and validate marketplace listings")
    p.add_argument("--marketplace", choices=["amazon", "walmart"], default="amazon")
    p.add_argument("--limit", type=int, default=0, help="Only process the first N products")
    p.set_defaults(func=cmd_listings)

    p = sub.add_parser("ask", help="Ask the support agent one question")
    p.add_argument("question")
    p.add_argument("--verbose", action="store_true", help="Show every tool call")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="Chat with the support agent")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("eval", help="Evaluate the support agent on the test questions")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(LLMClient(), args)


if __name__ == "__main__":
    main()
