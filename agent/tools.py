"""Tools the support agent can call, plus their JSON schemas for function calling.

Each tool reads the sample product catalog (data/products.csv) and order data
(data/orders.json). In a real store these would call the marketplace or
order-management APIs instead.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"


def load_products(path=DATA_DIR / "products.csv") -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["price_usd"] = float(row["price_usd"])
        row["stock"] = int(row["stock"])
        row["weight_lb"] = float(row["weight_lb"])
        row["features"] = [x.strip() for x in row["features"].split(";") if x.strip()]
    return rows


def load_orders(path=DATA_DIR / "orders.json") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


PRODUCTS = load_products()
ORDERS = load_orders()


def _find_product(sku: str):
    sku = sku.strip().upper()
    return next((p for p in PRODUCTS if p["sku"] == sku), None)


# ---------------------------------------------------------------- tools


def search_products(query: str, max_price: float | None = None, category: str | None = None) -> dict:
    """Keyword search over name, category, material and color."""
    words = [w for w in query.lower().split() if len(w) > 2]
    results = []
    for p in PRODUCTS:
        text = f"{p['product_name']} {p['category']} {p['material']} {p['color']}".lower()
        score = sum(1 for w in words if w in text)
        if words and score == 0:
            continue
        if max_price is not None and p["price_usd"] > max_price:
            continue
        if category and category.lower() not in p["category"].lower():
            continue
        results.append((score, p))
    results.sort(key=lambda x: -x[0])
    return {
        "count": len(results),
        "products": [
            {
                "sku": p["sku"],
                "name": p["product_name"],
                "category": p["category"],
                "price_usd": p["price_usd"],
                "in_stock": p["stock"] > 0,
                "stock": p["stock"],
            }
            for _, p in results[:5]
        ],
    }


def get_product_details(sku: str) -> dict:
    p = _find_product(sku)
    if not p:
        return {"error": f"No product found with SKU {sku}"}
    return p


def check_inventory(sku: str) -> dict:
    p = _find_product(sku)
    if not p:
        return {"error": f"No product found with SKU {sku}"}
    status = "in stock" if p["stock"] > 5 else "low stock" if p["stock"] > 0 else "out of stock"
    return {"sku": p["sku"], "name": p["product_name"], "units_available": p["stock"], "status": status}


def get_order_status(order_id: str) -> dict:
    order_id = order_id.strip().upper()
    order = next((o for o in ORDERS if o["order_id"] == order_id), None)
    if not order:
        return {"error": f"No order found with ID {order_id}"}
    product = _find_product(order["sku"])
    return {**order, "product_name": product["product_name"] if product else None}


def create_support_ticket(order_id: str, issue: str, priority: str = "normal") -> dict:
    """Logs a customer complaint to output/tickets.json (stands in for a helpdesk API)."""
    order = get_order_status(order_id)
    if "error" in order:
        return order
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / "tickets.json"
    tickets = json.loads(path.read_text()) if path.exists() else []
    ticket = {
        "ticket_id": f"TCK-{1001 + len(tickets)}",
        "order_id": order["order_id"],
        "issue": issue,
        "priority": priority if priority in ("low", "normal", "high") else "normal",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "open",
    }
    tickets.append(ticket)
    path.write_text(json.dumps(tickets, indent=2))
    return ticket


TOOL_FUNCTIONS = {
    "search_products": search_products,
    "get_product_details": get_product_details,
    "check_inventory": check_inventory,
    "get_order_status": get_order_status,
    "create_support_ticket": create_support_ticket,
}


def _schema(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOL_SCHEMAS = [
    _schema(
        "search_products",
        "Search the product catalog by keywords. Use when the customer describes a product by name or type.",
        {
            "query": {"type": "string", "description": "Keywords, e.g. 'coffee table' or 'rattan basket'"},
            "max_price": {"type": "number", "description": "Optional maximum price in USD"},
            "category": {"type": "string", "description": "Optional category, e.g. 'Living Room Furniture'"},
        },
        ["query"],
    ),
    _schema(
        "get_product_details",
        "Get full details (material, dimensions, features, price) for one product by SKU.",
        {"sku": {"type": "string", "description": "Product SKU, e.g. HD-1001"}},
        ["sku"],
    ),
    _schema(
        "check_inventory",
        "Check how many units of a product are available, by SKU.",
        {"sku": {"type": "string", "description": "Product SKU, e.g. HD-1001"}},
        ["sku"],
    ),
    _schema(
        "get_order_status",
        "Get status, carrier, tracking number and expected delivery date for an order.",
        {"order_id": {"type": "string", "description": "Order ID, e.g. ORD-5001"}},
        ["order_id"],
    ),
    _schema(
        "create_support_ticket",
        "Open a support ticket when a customer reports a problem (damaged, missing or wrong item, refund request).",
        {
            "order_id": {"type": "string", "description": "Order ID, e.g. ORD-5001"},
            "issue": {"type": "string", "description": "Short summary of the customer's problem"},
            "priority": {"type": "string", "enum": ["low", "normal", "high"]},
        },
        ["order_id", "issue"],
    ),
]


def run_tool(name: str, arguments: dict) -> dict:
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return func(**arguments)
    except TypeError as exc:  # wrong or missing arguments from the model
        return {"error": f"Bad arguments for {name}: {exc}"}
