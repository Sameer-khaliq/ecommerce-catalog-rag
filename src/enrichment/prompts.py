import json

GROUNDING_RULE = (
    "Grounding rule: only use facts present in the raw listing, the reference listings, "
    "or the schema below. Never invent specifications, certifications, or brand names that "
    "aren't supported by this context. If a field can't be determined, mark it as \"unknown\" "
    "and explain why in confidence_notes."
)


def build_attribute_prompt(raw_listing: dict, schema: dict, references: list[dict]) -> str:
    required = schema.get("required_attributes", [])
    optional = schema.get("optional_attributes", [])
    ref_examples = json.dumps(references, indent=2)

    return f"""You are completing missing product attributes for an e-commerce listing.

{GROUNDING_RULE}

Raw listing:
{json.dumps(raw_listing, indent=2)}

Required attributes for this category: {required}
Optional attributes (include if you can determine them): {optional}

Reference listings from the same category (style/format guidance only, not facts about THIS product):
{ref_examples}

Return ONLY a JSON object with two keys:
- "attributes": object with as many required attributes filled in as you can honestly support
- "confidence_notes": object mapping any inferred (not explicitly stated) attribute name to a short reason
"""


def build_description_prompt(raw_listing: dict, attributes: dict, style_guide: str | None = None) -> str:
    style_block = f"\nCategory style guide (follow this tone):\n{style_guide}\n" if style_guide else ""

    return f"""Write a polished, factual product description for an e-commerce listing.

{GROUNDING_RULE}
{style_block}
Title: {raw_listing.get('title', '')}
Category: {raw_listing.get('category', '')}
Known attributes: {json.dumps(attributes, indent=2)}

Write 2-3 sentences. Do not mention attributes that aren't in the list above.
Return ONLY a JSON object: {{"description": "..."}}
"""


def build_seo_tags_prompt(title: str, description: str, category: str, style_guide: str | None = None) -> str:
    style_block = f"\nCategory style guide (match this tone/vocabulary):\n{style_guide}\n" if style_guide else ""

    return f"""Generate 3-5 SEO search tags for this product.
{style_block}
Title: {title}
Category: {category}
Description: {description}

Return ONLY a JSON object: {{"seo_tags": ["tag1", "tag2", ...]}}
"""