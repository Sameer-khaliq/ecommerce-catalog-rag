import pytest

from src.ingestion.chunking import (
    chunk_attribute_table,
    chunk_prose,
    chunk_product,
    chunk_all_products,
    build_chunk_metadata,
)


@pytest.fixture
def complete_product():
    return {
        "product_id": "elec_001",
        "title": "Wireless Noise Cancelling Headphones",
        "category": "Electronics",
        "brand": "SoundCore",
        "price": 89.99,
        "description": "Over-ear headphones with 30hr battery life.",
        "attributes": {"color": "black", "battery_life": "30hrs", "connectivity": "Bluetooth 5.0"},
    }


@pytest.fixture
def sparse_product():
    return {
        "product_id": "elec_sparse_001",
        "category": "Electronics",
        "title": "bluetooth earbuds",
        "raw_attributes": {},
        "price": None,
    }


class TestAttributeTableChunk:
    def test_contains_all_attributes(self, complete_product):
        chunk = chunk_attribute_table(complete_product)
        for key, value in complete_product["attributes"].items():
            assert key in chunk and str(value) in chunk

    def test_contains_category(self, complete_product):
        chunk = chunk_attribute_table(complete_product)
        assert complete_product["category"] in chunk

    def test_contains_brand_when_present(self, complete_product):
        chunk = chunk_attribute_table(complete_product)
        assert complete_product["brand"] in chunk

    def test_missing_brand_does_not_crash(self, sparse_product):
        chunk = chunk_attribute_table(sparse_product)
        assert "Brand:" not in chunk  # brand line should be skipped entirely, not "Brand: Unknown"

    def test_empty_attributes_does_not_crash(self, sparse_product):
        chunk = chunk_attribute_table(sparse_product)
        assert "No attributes listed." in chunk

    def test_reads_raw_attributes_key(self):
        product = {"title": "test", "category": "Home", "raw_attributes": {"material": "steel"}}
        chunk = chunk_attribute_table(product)
        assert "material: steel" in chunk


class TestProseChunk:
    def test_not_empty(self, sparse_product):
        chunk = chunk_prose(sparse_product)
        assert len(chunk.strip()) > 0

    def test_null_price_does_not_print_none(self, sparse_product):
        chunk = chunk_prose(sparse_product)
        assert "None" not in chunk
        assert "price not listed" in chunk

    def test_contains_all_attribute_values(self, complete_product):
        chunk = chunk_prose(complete_product)
        for value in complete_product["attributes"].values():
            assert str(value) in chunk

    def test_missing_description_does_not_crash(self, sparse_product):
        assert chunk_prose(sparse_product)  # should not raise


class TestChunkMetadata:
    def test_metadata_has_required_fields(self, complete_product):
        meta = build_chunk_metadata(complete_product, "prose")
        assert meta["product_id"] == "elec_001"
        assert meta["category"] == "Electronics"
        assert meta["chunk_strategy"] == "prose"

    def test_null_price_defaults_to_zero(self, sparse_product):
        meta = build_chunk_metadata(sparse_product, "prose")
        assert meta["price"] == 0


class TestChunkProduct:
    def test_returns_two_chunks(self, complete_product):
        chunks = chunk_product(complete_product)
        assert len(chunks) == 2
        strategies = {c["metadata"]["chunk_strategy"] for c in chunks}
        assert strategies == {"attribute_table", "prose"}

    def test_no_chunk_exceeds_reasonable_length(self, complete_product):
        for chunk in chunk_product(complete_product):
            assert len(chunk["text"]) < 2000


class TestChunkAllProducts:
    def test_flattens_correctly(self, complete_product, sparse_product):
        chunks = chunk_all_products([complete_product, sparse_product])
        assert len(chunks) == 4  # 2 products x 2 strategies

    def test_empty_catalog_returns_empty_list(self):
        assert chunk_all_products([]) == []