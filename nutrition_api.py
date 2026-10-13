"""
nutrition_api.py
-----------------
Wrapper around the USDA FoodData Central (FDC) API — a completely free,
official U.S. government nutrition database. No credit card, no billing.

Get a free API key in 30 seconds here:
https://fdc.nal.usda.gov/api-key-signup.html

If you don't want to sign up right away, USDA also allows a shared
"DEMO_KEY" for light testing (rate-limited to ~30 requests/hour), which
this module falls back to automatically.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.nal.usda.gov/fdc/v1"
API_KEY = os.getenv("USDA_API_KEY", "DEMO_KEY")


class NutritionAPIError(Exception):
    """Raised when the USDA API call fails or returns no usable data."""
    pass


def search_food(query: str, page_size: int = 5) -> list[dict]:
    """
    Search USDA FoodData Central for foods matching `query`.
    Returns a list of simplified food dicts:
        {"fdcId": ..., "description": ..., "brandOwner": ...}
    """
    if not query or not query.strip():
        raise NutritionAPIError("Search query cannot be empty.")

    url = f"{BASE_URL}/foods/search"
    params = {
        "query": query,
        "pageSize": page_size,
        "api_key": API_KEY,
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise NutritionAPIError(f"Could not reach USDA API: {e}") from e

    data = response.json()
    foods = data.get("foods", [])

    if not foods:
        raise NutritionAPIError(f"No results found for '{query}'.")

    return [
        {
            "fdcId": f.get("fdcId"),
            "description": f.get("description"),
            "brandOwner": f.get("brandOwner", ""),
            "dataType": f.get("dataType", ""),
        }
        for f in foods
    ]


def get_food_details(fdc_id: int) -> dict:
    """
    Fetch full nutrient details for a specific food by its FDC id.
    Returns a dict with description + a clean list of key nutrients.
    """
    url = f"{BASE_URL}/food/{fdc_id}"
    params = {"api_key": API_KEY}

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise NutritionAPIError(f"Could not reach USDA API: {e}") from e

    data = response.json()

    # Nutrients we care most about for a "food nutrition analyzer"
    key_nutrients = {
        "Energy": "kcal",
        "Protein": "g",
        "Total lipid (fat)": "g",
        "Carbohydrate, by difference": "g",
        "Fiber, total dietary": "g",
        "Sugars, total including NLEA": "g",
        "Sodium, Na": "mg",
        "Calcium, Ca": "mg",
        "Iron, Fe": "mg",
        "Vitamin C, total ascorbic acid": "mg",
        "Cholesterol": "mg",
    }

    nutrients_found = {}
    for n in data.get("foodNutrients", []):
        name = n.get("nutrient", {}).get("name") or n.get("nutrientName")
        amount = n.get("amount")
        if name in key_nutrients and amount is not None:
            nutrients_found[name] = {"amount": amount, "unit": key_nutrients[name]}

    return {
        "description": data.get("description", "Unknown food"),
        "fdcId": fdc_id,
        "nutrients": nutrients_found,
    }


if __name__ == "__main__":
    # Quick manual test: python nutrition_api.py
    results = search_food("banana")
    print("Search results:", results)
    if results:
        details = get_food_details(results[0]["fdcId"])
        print("Details:", details)