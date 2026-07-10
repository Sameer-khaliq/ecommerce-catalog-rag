"""
One-off data generation script.

Uses Groq API (LLM) to generate, per category:
  1. Attribute schema        -> data/knowledge_base/{category}_attribute_schema.json
  2. Brand tone guide         -> data/knowledge_base/{category}_brand_guide.md
  3. Reference listings (18)  -> data/knowledge_base/{category}_reference_listings.json
  4. Sparse listings (10)     -> merged into data/raw/sparse_listings.json

Design note: reference listings and sparse listings describe DIFFERENT
products (not the same product stripped down). This forces retrieval to
generalize from similar-but-not-identical reference docs, rather than
"cheating" by matching an exact source product. That's what makes the
recall/faithfulness eval meaningful later (Day 31).

Run:
    python scripts/generate_data.py
"""

import json
import time
from pathlib import Path

from groq import Groq

from src.config import settings

client = Groq(api_key=settings.groq_api_key)

CATEGORIES = ["Electronics", "Fashion", "Beauty", "Home"]

KB_DIR = Path("data/knowledge_base")
RAW_DIR = Path("data/raw")
KB_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)


def call_groq(system_prompt: str, user_prompt: str, max_retries: int = 3) -> str:
    """Call Groq chat completion with simple retry/backoff."""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=settings.groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                response_format={"type": "json_object"} if "JSON" in system_prompt else None,
            )
            return response.choices[0].message.content
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = 2 ** attempt
            print(f"  [retry {attempt}/{max_retries}] Groq call failed: {e}. Waiting {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Groq call failed after {max_retries} retries: {last_err}")


def parse_json_safely(raw_text: str, context: str):
    """Parse JSON, stripping markdown code fences if the model added them."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"  [WARN] Failed to parse JSON for {context}: {e}")
        print(f"  Raw output: {raw_text[:300]}...")
        raise


def generate_attribute_schema(category: str) -> dict:
    system = (
        "You are an e-commerce catalog data architect. "
        "Respond ONLY with valid JSON. No preamble, no markdown fences."
    )
    user = f"""Generate an attribute schema for the "{category}" e-commerce category.

Return JSON with this exact structure:
{{
  "category": "{category}",
  "required_attributes": ["attr1", "attr2", ...],   // 6-8 attributes essential for this category
  "optional_attributes": ["attr1", "attr2", ...]     // 4-6 nice-to-have attributes
}}

Attributes should be realistic fields a seller would fill in for this category
(e.g. for Electronics: brand, connectivity, battery_life; for Fashion: material, fit, size_range).
"""
    raw = call_groq(system, user)
    return parse_json_safely(raw, f"{category} attribute schema")


def generate_brand_guide(category: str) -> str:
    system = (
        "You are a brand voice and content strategist for an e-commerce marketplace."
    )
    user = f"""Write a short brand tone guide (150-250 words) for product descriptions
in the "{category}" category. Format as markdown with these sections:
- Tone (2-3 adjectives, explained)
- Do's (3 bullet points)
- Don'ts (3 bullet points)
- One example sentence showing the tone in action

Keep it practical and specific to {category}, not generic marketing fluff.
"""
    return call_groq(system, user)


def generate_reference_listings(category: str, schema: dict, count: int = 18) -> list:
    system = (
        "You are an e-commerce content writer generating realistic, COMPLETE product listings. "
        "Respond ONLY with valid JSON. No preamble, no markdown fences."
    )
    user = f"""Generate {count} realistic, COMPLETE e-commerce product listings for the
"{category}" category. Each must be a distinct, plausible product (vary brand, price range, style).

Required attributes to fill for each product: {schema['required_attributes']}
Optional attributes to include where relevant: {schema['optional_attributes']}

Return a JSON array where each item has this structure:
{{
  "product_id": "string (e.g. elec_ref_001)",
  "category": "{category}",
  "title": "string",
  "description": "string (2-4 sentences, complete and well-written)",
  "attributes": {{ ...all required + some optional attributes filled in... }},
  "seo_tags": ["tag1", "tag2", "tag3"],
  "price": number
}}

Return ONLY the JSON array, nothing else.
"""
    raw = call_groq(system, user)
    return parse_json_safely(raw, f"{category} reference listings")


def generate_sparse_listings(category: str, count: int = 10) -> list:
    system = (
        "You are simulating REAL, LAZY e-commerce sellers who post incomplete product listings. "
        "Respond ONLY with valid JSON. No preamble, no markdown fences."
    )
    user = f"""Generate {count} realistic SPARSE/INCOMPLETE product listings for the
"{category}" category, as if posted by rushed sellers. These must be DIFFERENT products
from any reference catalog - just plausible new items in this category.

Vary the sparsity level:
- Some have only a title
- Some have a title + 1 raw attribute (e.g. just "color")
- Some have a title + price but no other detail
- A couple should be near-empty (title only, very vague, e.g. "black shoes size 9")

Return a JSON array where each item has this structure:
{{
  "product_id": "string (e.g. elec_sparse_001)",
  "category": "{category}",
  "title": "string (short, incomplete, realistic seller-typed text)",
  "raw_attributes": {{ ...0-2 attributes only... }},
  "price": number or null
}}

Return ONLY the JSON array, nothing else.
"""
    raw = call_groq(system, user)
    return parse_json_safely(raw, f"{category} sparse listings")


def main():
    all_sparse_listings = []

    for category in CATEGORIES:
        print(f"\n=== Generating data for: {category} ===")

        print("  -> attribute schema...")
        schema = generate_attribute_schema(category)
        (KB_DIR / f"{category.lower()}_attribute_schema.json").write_text(
            json.dumps(schema, indent=2), encoding="utf-8"
        )

        print("  -> brand guide...")
        guide = generate_brand_guide(category)
        (KB_DIR / f"{category.lower()}_brand_guide.md").write_text(guide, encoding="utf-8")

        print("  -> reference listings (18)...")
        references = generate_reference_listings(category, schema, count=18)
        (KB_DIR / f"{category.lower()}_reference_listings.json").write_text(
            json.dumps(references, indent=2), encoding="utf-8"
        )

        print("  -> sparse listings (10)...")
        sparse = generate_sparse_listings(category, count=10)
        all_sparse_listings.extend(sparse)

        print(f"  Done: {category} "
              f"({len(schema.get('required_attributes', []))} req attrs, "
              f"{len(references)} reference listings, {len(sparse)} sparse listings)")

    (RAW_DIR / "sparse_listings.json").write_text(
        json.dumps(all_sparse_listings, indent=2), encoding="utf-8"
    )

    print(f"\n=== DONE ===")
    print(f"Total sparse listings generated: {len(all_sparse_listings)}")
    print(f"Knowledge base files in: {KB_DIR}")
    print(f"Sparse listings file: {RAW_DIR / 'sparse_listings.json'}")


if __name__ == "__main__":
    main()