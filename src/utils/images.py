"""
Mapping of product categories to dynamic image URLs.
Uses the Unsplash API for high-end photography if configured.
Falls back to LoremFlickr for reliable placeholders if no key is provided.
"""
import urllib.request
import urllib.parse
import json
import logging
from src.config import UNSPLASH_ACCESS_KEY

logger = logging.getLogger(__name__)

# Dictionary cache to avoid hitting the Unsplash API rate limit (50/hr free tier)
IMAGE_CACHE = {}

def get_fallback_image(category_id: str) -> str:
    """Returns a reliable LoremFlickr URL using a keyword map."""
    keyword_map = {
        "agro_industry_and_commerce": "tractor",
        "air_conditioning": "airconditioner",
        "art": "painting",
        "audio": "headphones",
        "auto": "car",
        "baby": "baby",
        "bed_bath_table": "bedroom",
        "books_general_interest": "books",
        "books_imported": "books",
        "books_technical": "books",
        "cd_dvd_musicals": "dvd",
        "cine_photo": "camera",
        "computers": "computer",
        "computers_accessories": "laptop",
        "consoles_games": "playstation",
        "construction_tools_construction": "tools",
        "cool_stuff": "gadget",
        "costruction_tools_garden": "garden",
        "diapers_and_hygiene": "hygiene",
        "drinks": "drinks",
        "electronics": "electronics",
        "fashio_female_clothing": "dress",
        "fashion_bags_accessories": "bag",
        "fashion_shoes": "shoes",
        "fashion_sport": "sporting",
        "flowers": "flowers",
        "food": "food",
        "furniture_bedroom": "bed",
        "furniture_decor": "decor",
        "furniture_living_room": "sofa",
        "garden_tools": "garden",
        "health_beauty": "makeup",
        "home_appliances": "appliance",
        "housewares": "kitchen",
        "luggage_accessories": "luggage",
        "music": "guitar",
        "musical_instruments": "piano",
        "office_furniture": "desk",
        "party_supplies": "party",
        "pc_gamer": "gamingpc",
        "perfumery": "perfume",
        "pet_shop": "dog",
        "small_appliances": "blender",
        "sports_leisure": "sports",
        "stationery": "notebook",
        "telephony": "smartphone",
        "toys": "toys",
        "watches_gifts": "watch",
    }
    
    keyword = keyword_map.get(category_id, "ecommerce")
    lock_id = hash(category_id) % 1000
    
    return f"https://loremflickr.com/800/400/{keyword}?lock={lock_id}"

def get_category_image(category_id: str) -> str:
    """
    Returns a highly relevant, premium image URL for the product category.
    Prioritizes Unsplash Search API and falls back to LoremFlickr.
    """
    if category_id in IMAGE_CACHE:
        return IMAGE_CACHE[category_id]

    # If no API key is provided, use the fallback generator
    if not UNSPLASH_ACCESS_KEY:
        IMAGE_CACHE[category_id] = get_fallback_image(category_id)
        return IMAGE_CACHE[category_id]

    try:
        # Format a clean, relevant search query
        query = category_id.replace("_", " ")
        if query == "cool stuff":
            query = "gadgets electronics"

        url = f"https://api.unsplash.com/search/photos?query={urllib.parse.quote(query)}&orientation=landscape&per_page=1"
        
        req = urllib.request.Request(url, headers={
            "Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}",
            "Accept-Version": "v1"
        })

        with urllib.request.urlopen(req, timeout=3) as response:
            data = json.loads(response.read().decode())
            
            if data["results"]:
                image_url = data["results"][0]["urls"]["regular"]
                IMAGE_CACHE[category_id] = image_url
                return image_url
            else:
                logger.warning(f"No Unsplash results for '{query}'. Using fallback.")

    except Exception as e:
        logger.error(f"Unsplash API check failed ({e}). Falling back.")
    
    # If API call failed or returned empty results, use fallback
    IMAGE_CACHE[category_id] = get_fallback_image(category_id)
    return IMAGE_CACHE[category_id]
