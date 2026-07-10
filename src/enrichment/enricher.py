import json

from groq import Groq

from src.config import get_settings
from src.logging_config import get_logger, log_with_context
from src.enrichment.prompts import build_attribute_prompt, build_description_prompt, build_seo_tags_prompt
from src.enrichment.schemas import EnrichedListing
from src.enrichment.resilience import call_with_resilience

logger = get_logger(__name__)

settings = get_settings()
def _get_groq_client() -> Groq:
    return Groq(api_key=settings.groq_api_key)


def _call_llm_json(client: Groq, prompt: str) -> dict:
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return json.loads(response.choices[0].message.content)


def _note_if_degraded(result: dict, confidence_notes: dict, stage: str) -> None:
    if result.get("_degraded"):
        confidence_notes[f"_{stage}_degraded"] = result.get("_reason", "unknown error")
def _ensure_confidence_notes_complete(attributes: dict, confidence_notes: dict) -> dict:
    """Defensive check: if the LLM marked an attribute as unknown but forgot to
    log a confidence note for it, backfill a default note instead of trusting
    the LLM to always remember. LLM output should never be the only safeguard."""
    for key, value in attributes.items():
        if str(value).lower() == "unknown" and key not in confidence_notes:
            confidence_notes[key] = "not present in raw listing; LLM could not infer"
    return confidence_notes

def enrich_listing(raw_listing: dict, context: dict) -> EnrichedListing:
    client = _get_groq_client()
    product_id = raw_listing.get("product_id", "unknown")
    style_guide = context.get("style_guide")
    confidence_notes: dict = {}

    attr_result = call_with_resilience(
        lambda: _call_llm_json(client, build_attribute_prompt(raw_listing, context["schema"], context["references"])),
        product_id=product_id, stage="attributes",
    )
    _note_if_degraded(attr_result, confidence_notes, "attributes")
    attributes = attr_result.get("attributes", {})
    confidence_notes.update(attr_result.get("confidence_notes", {}))
    confidence_notes = _ensure_confidence_notes_complete(attributes, confidence_notes)

    desc_result = call_with_resilience(
        lambda: _call_llm_json(client, build_description_prompt(raw_listing, attributes, style_guide)),
        product_id=product_id, stage="description",
    )
    _note_if_degraded(desc_result, confidence_notes, "description")
    description = desc_result.get("description", raw_listing.get("title", ""))

    seo_result = call_with_resilience(
        lambda: _call_llm_json(client, build_seo_tags_prompt(raw_listing.get("title", ""), description, raw_listing.get("category", ""), style_guide)),
        product_id=product_id, stage="seo_tags",
    )
    _note_if_degraded(seo_result, confidence_notes, "seo_tags")
    seo_tags = seo_result.get("seo_tags") or [raw_listing.get("category", "product")]

    listing = EnrichedListing(
        product_id=product_id,
        category=raw_listing.get("category", "Unknown"),
        title=raw_listing.get("title", ""),
        description=description,
        attributes=attributes,
        seo_tags=seo_tags,
        price=raw_listing.get("price"),
        confidence_notes=confidence_notes,
    )
    log_with_context(logger, "info", "listing enriched", product_id=product_id, attribute_count=len(attributes))
    return listing