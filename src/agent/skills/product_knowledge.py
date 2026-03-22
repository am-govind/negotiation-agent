"""
Product Knowledge Skill: Retrieves real product data from the dataset.

Instead of the LLM making up product facts, this skill queries
actual data to provide accurate category stats, review scores,
and demand information.
"""
import pandas as pd

from src.agent.skills.registry import Skill, SkillRegistry
from src.config import FEATURES_PARQUET


_product_data: pd.DataFrame | None = None


def _load_data() -> pd.DataFrame:
    """Lazy-load product data."""
    global _product_data
    if _product_data is None:
        _product_data = pd.read_parquet(FEATURES_PARQUET)
    return _product_data


def _get_product_info(product_category: str, **kwargs) -> dict:
    """Retrieve real product statistics from the dataset."""
    df = _load_data()
    cat_data = df[df["product_category_name_english"] == product_category]

    if cat_data.empty:
        return {
            "found": False,
            "message": f"No data found for category: {product_category}",
            "available_categories": sorted(df["product_category_name_english"].unique().tolist())[:20],
        }

    return {
        "found": True,
        "category": product_category,
        "total_sales": len(cat_data),
        "price_stats": {
            "mean": round(cat_data["price"].mean(), 2),
            "median": round(cat_data["price"].median(), 2),
            "min": round(cat_data["price"].min(), 2),
            "max": round(cat_data["price"].max(), 2),
            "std": round(cat_data["price"].std(), 2),
        },
        "avg_review_score": round(cat_data["avg_review_score"].mean(), 2),
        "avg_review_count": round(cat_data["review_count"].mean(), 1),
        "avg_freight_value": round(cat_data["freight_value"].mean(), 2),
        "avg_weight_g": round(cat_data["product_weight_g"].mean(), 0),
        "demand_level": (
            "high" if len(cat_data) > 3000 else
            "medium" if len(cat_data) > 1000 else
            "low"
        ),
        "talking_points": _generate_talking_points(cat_data, product_category),
    }


def _generate_talking_points(cat_data: pd.DataFrame, category: str) -> list[str]:
    """Generate data-backed selling points."""
    points = []
    avg_review = cat_data["avg_review_score"].mean()
    total_sales = len(cat_data)

    if avg_review >= 4.0:
        points.append(f"Highly rated category with {avg_review:.1f}/5 average customer rating")
    if total_sales > 2000:
        points.append(f"Popular choice — over {total_sales:,} units sold on our platform")
    if cat_data["freight_value"].mean() < 20:
        points.append("Affordable shipping costs for this product category")

    high_review_pct = (cat_data["avg_review_score"] >= 4.0).mean() * 100
    if high_review_pct > 60:
        points.append(f"{high_review_pct:.0f}% of customers rated this category 4+ stars")

    if not points:
        points.append(f"Available in the {category.replace('_', ' ')} category")

    return points


def _compare_to_category_avg(
    product_category: str,
    current_price: float,
    **kwargs,
) -> dict:
    """Compare a price to the category average."""
    df = _load_data()
    cat_data = df[df["product_category_name_english"] == product_category]

    if cat_data.empty:
        return {"error": f"No data for category: {product_category}"}

    avg_price = cat_data["price"].mean()
    median_price = cat_data["price"].median()
    pct_diff_mean = ((current_price - avg_price) / avg_price) * 100
    pct_diff_median = ((current_price - median_price) / median_price) * 100

    # What percentile is this price at?
    percentile = (cat_data["price"] <= current_price).mean() * 100

    return {
        "current_price": current_price,
        "category_avg": round(avg_price, 2),
        "category_median": round(median_price, 2),
        "vs_avg_pct": round(pct_diff_mean, 1),
        "vs_median_pct": round(pct_diff_median, 1),
        "price_percentile": round(percentile, 1),
        "assessment": (
            f"At ${current_price:.2f}, this is in the {percentile:.0f}th percentile "
            f"for {product_category.replace('_', ' ')} "
            f"({'above' if pct_diff_mean > 0 else 'below'} the ${avg_price:.2f} average)."
        ),
    }


def register_product_skills(registry: SkillRegistry):
    """Register product knowledge skills."""

    registry.register(Skill(
        name="get_product_info",
        description=(
            "Retrieve real product statistics from the sales database: "
            "price range, review scores, demand level, and data-backed "
            "talking points. Use this to make factual claims about the product."
        ),
        category="product",
        execute=_get_product_info,
        tool_schema={
            "type": "function",
            "function": {
                "name": "get_product_info",
                "description": (
                    "Get real product statistics and selling points for a category. "
                    "Use this instead of making up product facts."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_category": {
                            "type": "string",
                            "description": "The product category to look up",
                        },
                    },
                    "required": ["product_category"],
                },
            },
        },
    ))

    registry.register(Skill(
        name="compare_to_category_avg",
        description=(
            "Compare a specific price to the category average and median. "
            "Returns the percentile ranking and assessment."
        ),
        category="product",
        execute=_compare_to_category_avg,
        tool_schema={
            "type": "function",
            "function": {
                "name": "compare_to_category_avg",
                "description": (
                    "Compare your current offer to the market average for this "
                    "category. Use this when the customer says your price is too high."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_category": {
                            "type": "string",
                            "description": "Product category",
                        },
                        "current_price": {
                            "type": "number",
                            "description": "The price to compare",
                        },
                    },
                    "required": ["product_category", "current_price"],
                },
            },
        },
    ))
