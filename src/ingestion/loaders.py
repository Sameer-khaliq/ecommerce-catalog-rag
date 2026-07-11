"""Reads a JSON file of product listings and returns them as a list of dicts.
    Works for both raw sparse listings and enriched listings"""

import json
from pathlib import Path

from src.logging_config import get_logger, log_with_context

logger = get_logger(__name__)

def load_listings(filepath:str) ->list[dict]:
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"file not found: {filepath}")
    
    with open (path, "r", encoding = "utf-8") as f:
        listings = json.load(f)
    
    if not isinstance (listings, list):
        raise ValueError(f"Expected a list , got{type(listings).__name__}")
    
    log_with_context(logger, "info", "Loaded listings from file", filepath = filepath, num_listings = len(listings))

    return listings
listings = load_listings("data/raw/sparse_listings.json")
print(len(listings), listings[0])