"""Rule-based guardrails that check an AI-written listing before it is accepted.

The limits below are simplified example rules for this project. Always check
the marketplace's current style guide before using them in production.
"""

from __future__ import annotations

import re

MARKETPLACE_RULES = {
    "amazon": {
        "title_max": 200,
        "bullet_count": 5,
        "bullet_max": 255,
        "description_min": 300,
        "description_max": 2000,
        "search_terms_max_bytes": 249,
    },
    "walmart": {
        "title_max": 75,
        "bullet_count": 5,
        "bullet_max": 200,
        "description_min": 150,
        "description_max": 4000,
        "search_terms_max_bytes": 249,
    },
}

# Claims marketplaces usually reject or that we cannot prove
BANNED_PHRASES = [
    "best seller", "bestseller", "#1", "number one", "free shipping", "guaranteed",
    "100%", "cheapest", "sale", "discount", "limited time", "eco-friendly", "non-toxic",
]


def validate_listing(listing: dict, product: dict, marketplace: str = "amazon") -> list[str]:
    """Returns a list of problems. An empty list means the listing passed."""
    rules = MARKETPLACE_RULES[marketplace]
    issues = []

    title = str(listing.get("title", "")).strip()
    bullets = listing.get("bullets") or []
    description = str(listing.get("description", "")).strip()
    search_terms = str(listing.get("search_terms", "")).strip()

    if not title:
        issues.append("Title is missing.")
    elif len(title) > rules["title_max"]:
        issues.append(f"Title has {len(title)} characters; the limit is {rules['title_max']}.")

    if not isinstance(bullets, list) or len(bullets) != rules["bullet_count"]:
        issues.append(f"Need exactly {rules['bullet_count']} bullet points.")
    else:
        for i, b in enumerate(bullets, 1):
            if len(str(b)) > rules["bullet_max"]:
                issues.append(f"Bullet {i} has {len(str(b))} characters; the limit is {rules['bullet_max']}.")

    if len(description) < rules["description_min"]:
        issues.append(f"Description is too short ({len(description)} chars, minimum {rules['description_min']}).")
    elif len(description) > rules["description_max"]:
        issues.append(f"Description is too long ({len(description)} chars, maximum {rules['description_max']}).")

    if len(search_terms.encode("utf-8")) > rules["search_terms_max_bytes"]:
        issues.append(f"Search terms exceed {rules['search_terms_max_bytes']} bytes.")

    full_text = " ".join([title, " ".join(map(str, bullets)), description]).lower()
    for phrase in BANNED_PHRASES:
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", full_text):
            issues.append(f"Remove the banned or unprovable claim: '{phrase}'.")

    # Key facts must come from the catalog, so the listing cannot leave them out
    if not any(w in full_text for w in _key_words(product["material"])):
        issues.append(f"Mention the material ({product['material']}).")
    if not any(w in full_text for w in _key_words(product["color"])):
        issues.append(f"Mention the color ({product['color']}).")
    if "dimension" not in full_text and product["dimensions_in"].split()[0] not in full_text:
        issues.append(f"Include the dimensions ({product['dimensions_in']} inches).")

    return issues


_FILLER = {"and", "with", "solid", "natural", "engineered", "base", "shade", "matte", "antique", "rustic"}


def _key_words(text: str) -> list[str]:
    """'Ceramic base and linen shade' -> ['ceramic', 'linen']"""
    return [w for w in text.lower().replace(",", " ").split() if w not in _FILLER]
