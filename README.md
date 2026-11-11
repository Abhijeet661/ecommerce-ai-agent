# E-commerce AI Agent: Listing Writer + Customer Support

![tests](https://github.com/Abhijeet661/ecommerce-ai-agent/actions/workflows/tests.yml/badge.svg)

A Python AI agent for an online home decor and furniture store. It automates two everyday e-commerce jobs:

1. **Listing agent**: turns raw catalog data (SKU, material, color, dimensions, features) into marketplace-ready listings for Amazon or Walmart: title, 5 bullet points, description and search terms. A rule-based validator checks every listing, and if it fails, the exact problems are sent back to the model so it can fix them (**self-correction loop**).
2. **Support agent**: answers customer questions about products, stock and orders using **LLM function calling**. The model decides which tool to call, the code runs it on real data, and the model answers from the result instead of guessing. It can also open a support ticket for damaged or wrong items.

It also has an **evaluation script** that measures how often the agent picks the right tool and gives the right facts, plus **unit tests** that run with a mock LLM (no API key needed).

## How it works

```
                       ┌──────────────────────────────┐
 Customer question ──▶ │  Support agent (LLM)         │
                       │  decides: answer or use tool │
                       └──────────────┬───────────────┘
                                      │ tool call (JSON)
          ┌───────────────┬───────────┼──────────────┬─────────────────────┐
          ▼               ▼           ▼              ▼                     ▼
   search_products  get_product   check_inventory  get_order_status  create_support_ticket
                    _details
          └───────────────┴───────────┬──────────────┴─────────────────────┘
                                      │ tool result (JSON) goes back to the LLM
                                      ▼
                           Final answer to customer


 Product facts ──▶ LLM writes listing ──▶ Validator ──pass──▶ Save (JSON + CSV)
                         ▲                   │
                         └──── issues ───────┘  (up to 3 attempts)
```

## Features

- Tool-calling agent loop with a step limit, bad-argument handling and a full trace of every tool call
- 5 tools over a sample catalog (10 products) and order data (6 orders)
- Guardrails for listings: title length, exactly 5 bullets, bullet length, description length, search-term byte limit, banned claims ("best seller", "free shipping", "guaranteed"...), and required facts (material, color, dimensions)
- Works with any OpenAI-compatible API: Google Gemini (free key), Groq, OpenAI
- Evaluation: tool accuracy, answer accuracy and average steps on 10 test questions
- 10 pytest tests using a fake LLM, so core logic is tested without API calls; they run automatically on every push with GitHub Actions

## Setup

```bash
git clone https://github.com/Abhijeet661/ecommerce-ai-agent.git
cd ecommerce-ai-agent
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then paste your API key into .env
```

Get a free Gemini API key from Google AI Studio (aistudio.google.com) and put it in `.env` as `LLM_API_KEY`. To use Groq instead, change the three values as shown in `.env.example`. If the default model name is retired, set `LLM_MODEL` to the current Flash model name shown in AI Studio.

## Usage

```bash
# Write and validate Amazon listings for the first 3 products
python main.py listings --marketplace amazon --limit 3

# Ask the support agent one question and show its tool calls
python main.py ask "Where is my order ORD-5002?" --verbose

# Chat with the support agent
python main.py chat --verbose

# Evaluate the agent on the test questions
python main.py eval --verbose

# Run unit tests (no API key needed)
python -m pytest -q
```

Results are saved in `output/` (`listings_amazon.json`, `listings_amazon.csv`, `eval_report.json`, `tickets.json`).

## Project structure

```
ecommerce-ai-agent/
├── main.py                  # command-line interface
├── agent/
│   ├── llm.py               # OpenAI-compatible API client with retry on rate limits
│   ├── tools.py             # 5 tools + JSON schemas for function calling
│   ├── support_agent.py     # tool-calling agent loop
│   ├── listing_agent.py     # listing writer with self-correction
│   ├── validators.py        # marketplace rules (guardrails)
│   └── evaluate.py          # tool accuracy and answer accuracy
├── data/
│   ├── products.csv         # sample catalog (home decor and furniture)
│   ├── orders.json          # sample orders
│   └── eval_questions.json  # 10 test questions with expected tools and facts
├── tests/test_agent.py      # pytest tests with a mock LLM
└── .github/workflows/       # runs the tests on every push (GitHub Actions)
```

## Notes

- Product and order data are sample data made for this project.
- Marketplace limits in `validators.py` are simplified example rules. Check each marketplace's current style guide before real use.

## Next steps

- Rebuild the support flow with LangGraph and add memory across sessions
- Add RAG over product manuals and return policies
- Connect real APIs (Shopify, Amazon SP-API) and trigger runs from n8n
- Add a Streamlit web interface
