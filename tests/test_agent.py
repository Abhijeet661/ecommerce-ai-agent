"""Tests run without an API key: a FakeLLM replays scripted model replies."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.listing_agent import generate_listing
from agent.llm import LLMResponse, ToolCall, parse_json_reply
from agent.support_agent import SupportAgent
from agent.tools import check_inventory, get_order_status, run_tool, search_products
from agent.validators import validate_listing


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, tools=None, temperature=0.2):
        self.calls.append([dict(m) for m in messages])
        return self.replies.pop(0)


PRODUCT = {
    "sku": "HD-1001", "product_name": "Mango Wood Coffee Table with Open Shelf",
    "category": "Living Room Furniture", "material": "Solid mango wood", "color": "Natural brown",
    "dimensions_in": "42 x 24 x 18", "weight_lb": 38.0, "price_usd": 289.99, "stock": 14,
    "features": ["Lower open shelf for storage"],
}

GOOD_LISTING = {
    "title": "Mango Wood Coffee Table with Open Shelf, Natural Brown, 42 x 24 x 18 Inches",
    "bullets": [
        "SOLID MANGO WOOD: Built from solid mango wood with a hand-finished grain.",
        "OPEN SHELF: A lower shelf keeps books and baskets within reach.",
        "RIGHT SIZE: Measures 42 x 24 x 18 inches for most living rooms.",
        "WARM COLOR: Natural brown tone suits rustic and modern decor.",
        "EASY SETUP: Simple assembly with the included tools.",
    ],
    "description": ("This coffee table is made from solid mango wood in a natural brown finish. "
                    "It measures 42 x 24 x 18 inches and has a lower open shelf for storage, so you can keep "
                    "magazines, remotes and baskets tidy. The hand-finished wood grain makes every piece "
                    "slightly different. Assembly is simple with the tools included in the box."),
    "search_terms": "living room centre table rustic farmhouse storage shelf sofa table",
}


# ------------------------------------------------------------ tools

def test_search_finds_product_and_respects_price():
    res = search_products("coffee table")
    assert res["products"][0]["sku"] == "HD-1001"
    cheap = search_products("coffee table", max_price=100)["products"]
    assert all(p["price_usd"] <= 100 for p in cheap)
    assert "HD-1001" not in [p["sku"] for p in cheap]


def test_inventory_status():
    assert check_inventory("HD-1003")["status"] == "out of stock"
    assert check_inventory("hd-1006")["status"] == "low stock"
    assert "error" in check_inventory("XX-0000")


def test_order_lookup_and_bad_arguments():
    assert get_order_status("ORD-5002")["carrier"] == "FedEx"
    assert "error" in run_tool("get_order_status", {"wrong_arg": 1})
    assert "error" in run_tool("delete_everything", {})


# ------------------------------------------------------------ validator

def test_good_listing_passes():
    assert validate_listing(GOOD_LISTING, PRODUCT, "amazon") == []


def test_validator_catches_problems():
    bad = dict(GOOD_LISTING, title="X" * 210, bullets=GOOD_LISTING["bullets"][:3],
               description="Best seller table with free shipping.")
    issues = " ".join(validate_listing(bad, PRODUCT, "amazon"))
    assert "Title has 210" in issues
    assert "exactly 5 bullet" in issues
    assert "free shipping" in issues and "best seller" in issues
    assert "too short" in issues


def test_walmart_title_limit_is_stricter():
    long_title = dict(GOOD_LISTING, title=GOOD_LISTING["title"] + " for Living Room")
    assert validate_listing(long_title, PRODUCT, "amazon") == []
    assert any("limit is 75" in i for i in validate_listing(long_title, PRODUCT, "walmart"))


def test_parse_json_with_code_fence():
    assert parse_json_reply('```json\n{"a": 1}\n```') == {"a": 1}


# ------------------------------------------------------------ agents

def test_support_agent_calls_tool_then_answers():
    llm = FakeLLM([
        LLMResponse(None, [ToolCall("c1", "get_order_status", '{"order_id": "ORD-5002"}')]),
        LLMResponse("Your order ORD-5002 has shipped with FedEx and should arrive by 2026-11-13."),
    ])
    result = SupportAgent(llm).ask("Where is my order ORD-5002?")
    assert result.steps == 2
    assert result.tool_calls[0]["tool"] == "get_order_status"
    assert result.tool_calls[0]["result"]["status"] == "Shipped"
    # The tool result must be sent back to the model on the second call
    tool_msg = llm.calls[1][-1]
    assert tool_msg["role"] == "tool" and json.loads(tool_msg["content"])["carrier"] == "FedEx"


def test_support_agent_stops_after_max_steps():
    loop = [LLMResponse(None, [ToolCall(f"c{i}", "check_inventory", '{"sku": "HD-1001"}')]) for i in range(3)]
    result = SupportAgent(FakeLLM(loop), max_steps=3).ask("stock?")
    assert result.steps == 3 and "human agent" in result.answer


def test_listing_agent_self_corrects():
    first = dict(GOOD_LISTING, title=GOOD_LISTING["title"] + " - Best Seller")
    llm = FakeLLM([LLMResponse(json.dumps(first)), LLMResponse(json.dumps(GOOD_LISTING))])
    result = generate_listing(llm, PRODUCT, "amazon")
    assert result.passed and result.attempts == 2
    # The second request must include the validator's feedback
    assert "best seller" in llm.calls[1][-1]["content"]
