from src.enrichment.prompts import build_attribute_prompt, build_description_prompt, build_seo_tags_prompt, GROUNDING_RULE
from src.enrichment.retriever import get_retrieval_context
from src.enrichment.resilience import call_with_resilience, CircuitBreaker
from src.enrichment.cache import get_cached, set_cached, _hash_listing

SAMPLE_SCHEMA = {"category": "Beauty", "required_attributes": ["brand", "product_type"], "optional_attributes": ["fragrance"]}
SAMPLE_REFERENCES = [{"product_id": "ref1", "category": "Beauty", "title": "Ref product"}]
SAMPLE_RAW = {"product_id": "beauty_005", "category": "Beauty", "title": "SPF lotion", "raw_attributes": {}, "price": 27.5}


class TestPrompts:
    def test_attribute_prompt_contains_grounding_rule(self):
        assert GROUNDING_RULE in build_attribute_prompt(SAMPLE_RAW, SAMPLE_SCHEMA, SAMPLE_REFERENCES)

    def test_attribute_prompt_contains_required_attributes(self):
        prompt = build_attribute_prompt(SAMPLE_RAW, SAMPLE_SCHEMA, SAMPLE_REFERENCES)
        assert "brand" in prompt and "product_type" in prompt

    def test_description_prompt_contains_grounding_rule(self):
        assert GROUNDING_RULE in build_description_prompt(SAMPLE_RAW, {"brand": "EcoShield"})

    def test_description_prompt_includes_style_guide_when_provided(self):
        prompt = build_description_prompt(SAMPLE_RAW, {"brand": "EcoShield"}, style_guide="Confident, Fresh tone")
        assert "Confident, Fresh tone" in prompt

    def test_seo_prompt_requests_json_only(self):
        assert "seo_tags" in build_seo_tags_prompt("title", "desc", "Beauty")


class TestRetrieverContext:
    def test_returns_correct_category_schema_and_references(self):
        kb = {
            "schemas": {"Beauty": SAMPLE_SCHEMA},
            "references": {"Beauty": SAMPLE_REFERENCES * 5},
            "brand_guides": {"Beauty": "## Tone\nConfident, Fresh, Inclusive"},
        }
        context = get_retrieval_context(SAMPLE_RAW, kb, max_references=2)
        assert context["schema"] == SAMPLE_SCHEMA
        assert len(context["references"]) == 2
        assert "Confident" in context["style_guide"]

    def test_unknown_category_returns_empty_schema(self):
        kb = {"schemas": {}, "references": {}, "brand_guides": {}}
        context = get_retrieval_context({"category": "Toys"}, kb)
        assert context["schema"] == {} and context["references"] == []
        assert context["style_guide"] is None

    def test_missing_style_guide_returns_none(self):
        kb = {"schemas": {"Beauty": {}}, "references": {"Beauty": []}, "brand_guides": {}}
        assert get_retrieval_context(SAMPLE_RAW, kb)["style_guide"] is None


class TestResilience:
    def test_succeeds_on_first_try(self):
        assert call_with_resilience(lambda: {"ok": True}, product_id="p1", stage="test") == {"ok": True}

    def test_retries_then_succeeds(self):
        attempts = {"count": 0}
        def flaky():
            attempts["count"] += 1
            if attempts["count"] < 2:
                raise ValueError("temporary failure")
            return {"ok": True}
        result = call_with_resilience(flaky, product_id="p1", stage="test", base_delay=0.01)
        assert result == {"ok": True} and attempts["count"] == 2

    def test_degrades_gracefully_after_max_retries(self):
        def always_fails():
            raise ValueError("permanent failure")
        result = call_with_resilience(always_fails, product_id="p1", stage="test", max_retries=2, base_delay=0.01)
        assert result.get("_degraded") is True

    def test_circuit_breaker_opens_after_threshold(self):
        breaker = CircuitBreaker(failure_threshold=2)
        breaker.record_failure()
        assert breaker.is_open is False
        breaker.record_failure()
        assert breaker.is_open is True

    def test_circuit_breaker_resets_on_success(self):
        breaker = CircuitBreaker(failure_threshold=2)
        breaker.record_failure()
        breaker.record_success()
        assert breaker.consecutive_failures == 0


class TestCache:
    def test_cache_miss_returns_none(self):
        assert get_cached({"product_id": "unique_test_1", "title": "x"}) is None

    def test_cache_hit_after_set(self):
        listing = {"product_id": "unique_test_2", "title": "y"}
        set_cached(listing, {"enriched": True})
        assert get_cached(listing) == {"enriched": True}

    def test_hash_is_stable_regardless_of_key_order(self):
        assert _hash_listing({"product_id": "p1", "title": "x"}) == _hash_listing({"title": "x", "product_id": "p1"})