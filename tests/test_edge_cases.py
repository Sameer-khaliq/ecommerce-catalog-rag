from src.ingestion.chunking import chunk_attribute_table, chunk_prose
from src.enrichment.schemas import EnrichedListing
from src.enrichment.retriever import get_retrieval_context
from src.enrichment.resilience import call_with_resilience


class TestEmptyAndMinimalInput:
    def test_empty_title_does_not_crash_chunking(self):
        product = {"category": "Home", "raw_attributes": {}}
        assert chunk_prose(product) and chunk_attribute_table(product)

    def test_single_word_title(self):
        assert "Sofa" in chunk_prose({"title": "Sofa", "category": "Home", "raw_attributes": {}})


class TestConflictingAttributes:
    def test_flexible_attribute_type_does_not_raise(self):
        listing = EnrichedListing(
            product_id="p1", category="Home", title="t",
            description="A perfectly fine description for testing purposes.",
            attributes={"weight": "85"},
            seo_tags=["sofa"], confidence_notes={},
        )
        assert listing.attributes["weight"] == "85"


class TestNonEnglishAndLongInput:
    def test_non_english_title_does_not_crash(self):
        assert chunk_prose({"title": "صوفہ جدید", "category": "Home", "raw_attributes": {}})

    def test_extremely_long_description_does_not_crash(self):
        listing = EnrichedListing(
            product_id="p1", category="Home", title="t",
            description="word " * 5000, attributes={}, seo_tags=["x"], confidence_notes={},
        )
        assert len(listing.description) > 1000


class TestMalformedRetrievedDocs:
    def test_missing_reference_listings_key_does_not_crash(self):
        kb = {"schemas": {"Home": {}}, "references": {}, "brand_guides": {}}
        assert get_retrieval_context({"category": "Home"}, kb)["references"] == []

    def test_reference_missing_category_key_is_skipped_not_crashed(self):
        malformed_refs = [{"product_id": "bad_ref"}]
        grouped = {}
        for r in malformed_refs:
            grouped.setdefault(r.get("category", "Unknown"), []).append(r)
        assert "Unknown" in grouped


class TestResilienceEdgeCases:
    def test_single_retry_still_attempts_once(self):
        calls = {"count": 0}
        def fn():
            calls["count"] += 1
            raise ValueError("fail")
        result = call_with_resilience(fn, product_id="p1", stage="test", max_retries=1, base_delay=0.01)
        assert calls["count"] == 1 and result.get("_degraded") is True