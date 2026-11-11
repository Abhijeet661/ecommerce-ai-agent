"""Listing agent: writes a marketplace listing, checks it, and fixes it until it passes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .llm import parse_json_reply
from .validators import MARKETPLACE_RULES, validate_listing

LISTING_PROMPT = """You are an e-commerce copywriter for a US home decor and furniture store.
Write a {marketplace} product listing using ONLY the facts below. Do not invent features, certifications or claims.

Product facts:
{facts}

Rules:
- Title: at most {title_max} characters. Format: Product type + key material + color + main feature.
- Exactly {bullet_count} bullet points, each at most {bullet_max} characters, each starting with a short benefit in capitals.
- Description: {description_min}-{description_max} characters, plain text, mention material, color and dimensions.
- search_terms: space-separated keywords shoppers would type, not repeating title words, under {search_terms_max_bytes} bytes.
- Do not use words like "best seller", "#1", "free shipping", "guaranteed", "sale", "discount" or "eco-friendly".

Reply with JSON only, in this shape:
{{"title": "...", "bullets": ["...", "...", "...", "...", "..."], "description": "...", "search_terms": "..."}}
"""

FIX_PROMPT = """Your listing failed these checks:
{issues}

Fix every problem and reply with the full corrected listing as JSON only, in the same shape."""


@dataclass
class ListingResult:
    sku: str
    listing: dict
    passed: bool
    attempts: int
    issues: list[str] = field(default_factory=list)


def generate_listing(llm, product: dict, marketplace: str = "amazon", max_attempts: int = 3) -> ListingResult:
    facts = json.dumps({k: v for k, v in product.items() if k != "stock"}, indent=2)
    prompt = LISTING_PROMPT.format(marketplace=marketplace.title(), facts=facts, **MARKETPLACE_RULES[marketplace])
    messages = [{"role": "user", "content": prompt}]

    listing, issues = {}, ["No listing generated."]
    for attempt in range(1, max_attempts + 1):
        reply = llm.chat(messages, temperature=0.4)
        messages.append({"role": "assistant", "content": reply.content or ""})
        try:
            listing = parse_json_reply(reply.content)
        except (ValueError, json.JSONDecodeError):
            issues = ["Reply was not valid JSON."]
        else:
            issues = validate_listing(listing, product, marketplace)

        if not issues:
            return ListingResult(product["sku"], listing, True, attempt)

        # Self-correction: send the exact problems back to the model
        messages.append({"role": "user", "content": FIX_PROMPT.format(issues="\n".join(f"- {i}" for i in issues))})

    return ListingResult(product["sku"], listing, False, max_attempts, issues)
