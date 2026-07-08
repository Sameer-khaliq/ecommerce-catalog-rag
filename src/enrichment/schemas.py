from pydantic import BaseModel, Field, field_validator


class EnrichedListing(BaseModel):
    product_id: str
    category: str
    title: str
    description: str
    attributes: dict[str, str | bool | int | float | list[str]]
    seo_tags: list[str]
    price: float | None = None

    # Maps attribute name -> reason it was inferred rather than stated,
    # e.g. {"brand": "no brand mentioned in raw title; inferred as Generic"}
    confidence_notes: dict[str, str] = Field(default_factory=dict)

    @field_validator("seo_tags")
    @classmethod
    def seo_tags_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("seo_tags must not be empty")
        return v

    @field_validator("description")
    @classmethod
    def description_not_too_short(cls, v: str) -> str:
        if len(v.strip()) < 20:
            raise ValueError("description too short — likely a failed generation")
        return v


def validate_required_attributes(listing: EnrichedListing, category_schema: dict) -> list[str]:
    """Checks the enriched listing against its category's required_attributes list
    (loaded from attribute_schemas.json). Returns missing attribute names — empty list means valid.
    """
    required = category_schema.get("required_attributes", [])
    missing = [attr for attr in required if attr not in listing.attributes]
    return missing