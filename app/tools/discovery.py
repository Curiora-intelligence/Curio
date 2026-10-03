"""Fictional demo catalogue and transparent, deterministic ranking."""
from dataclasses import asdict, dataclass
from math import asin, cos, radians, sin, sqrt
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.tools.registry import Permission, Tool


class DiscoveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    category: str = Field(min_length=1, max_length=80)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    budget_max: float | None = Field(default=None, gt=0)
    preferences: list[str] = Field(default_factory=list, max_length=12)
    availability: Literal["today", "any"] = "any"

    @model_validator(mode="after")
    def coordinates_together(self):
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("Provide both latitude and longitude, or neither.")
        if any(len(p) > 200 for p in self.preferences):
            raise ValueError("Each preference must be at most 200 characters.")
        return self


@dataclass(frozen=True)
class Provider:
    id: str
    name: str
    category: str
    rating: float
    latitude: float
    longitude: float
    price: float
    available_today: bool
    tags: tuple[str, ...]
    fictional: bool = True
    currency: str = "INR"


CATALOGUE = (
    Provider("demo-food-1", "Demo Saffron Kitchen (fictional)", "food", 4.6, 17.4401, 78.3489, 280, True, ("spicy", "chicken", "biryani")),
    Provider("demo-food-2", "Demo Green Bowl (fictional)", "food", 4.8, 17.445, 78.355, 240, True, ("vegetarian", "biryani", "mild")),
    Provider("demo-food-3", "Demo Royal Pot (fictional)", "food", 4.9, 17.461, 78.369, 420, False, ("chicken", "biryani", "spicy")),
    Provider("demo-plumber-1", "Demo Tap Repair (fictional)", "plumber", 4.7, 17.442, 78.349, 450, True, ("tap", "leak", "plumbing")),
    Provider("demo-plumber-2", "Demo Pipe Care (fictional)", "plumber", 4.9, 17.452, 78.366, 650, False, ("tap", "pipe", "plumbing")),
    Provider("demo-shopping-1", "Demo Minimal Store (fictional)", "shopping", 4.5, 17.440, 78.350, 2200, True, ("minimal", "black", "products")),
)
WEIGHTS = {"rating": .30, "distance": .25, "budget_fit": .20, "availability": .15, "preference_match": .10}
ALIASES = {"biryani": "food", "restaurant": "food", "restaurants": "food", "plumbing": "plumber", "repair": "plumber"}


def distance_km(lat1, lon1, lat2, lon2) -> float:
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 6371 * 2 * asin(sqrt(min(1, a)))


def search_services(request: DiscoveryRequest) -> dict:
    category = ALIASES.get(request.category.strip().lower(), request.category.strip().lower())
    recommendations = []
    for provider in CATALOGUE:
        if provider.category != category:
            continue
        # Explicit ceilings and same-day requests are hard constraints as well as score inputs.
        if request.budget_max is not None and provider.price > request.budget_max:
            continue
        if request.availability == "today" and not provider.available_today:
            continue
        distance = None if request.latitude is None else distance_km(request.latitude, request.longitude, provider.latitude, provider.longitude)
        import re
        desired = set(re.findall(r"[a-z]+", " ".join(request.preferences).lower()))
        matched = sorted(desired & set(provider.tags))
        components = {
            "rating": provider.rating / 5,
            "distance": max(0, 1 - distance / 20) if distance is not None else .5,
            "budget_fit": 1.0 if request.budget_max is not None else .5,
            "availability": 1.0 if provider.available_today else 0,
            "preference_match": len(matched) / len(desired) if desired else .5,
        }
        reasons = [f"Fictional demo rating: {provider.rating}/5.",
                   f"{distance:.2f} km from supplied coordinates." if distance is not None else "Distance unknown; supply coordinates for nearby ranking.",
                   f"Demo price ₹{provider.price:g} fits your ₹{request.budget_max:g} ceiling." if request.budget_max else f"Demo price ₹{provider.price:g}; no budget supplied.",
                   "Available today in the fictional catalogue." if provider.available_today else "Not available today in the fictional catalogue.",
                   "Preference matches: " + (", ".join(matched) or "none specified or matched")]
        recommendations.append({**asdict(provider), "distance_km": round(distance, 3) if distance is not None else None,
                                "score": round(sum(WEIGHTS[k] * v for k, v in components.items()), 6),
                                "components": components, "reasons": reasons})
    recommendations.sort(key=lambda item: (-item["score"], item["id"]))
    return {"source": "fictional_demo_catalogue", "fictional": True, "weights": WEIGHTS,
            "location_required": request.latitude is None, "recommendations": recommendations,
            "notice": "All providers, prices, ratings and availability are fictional. No booking is performed."}


class ExplainRequest(DiscoveryRequest):
    provider_id: str = Field(min_length=1, max_length=80)


def register_discovery_tools(registry):
    async def search(args, context):
        return search_services(args)

    async def explain(args, context):
        result = search_services(DiscoveryRequest(**args.model_dump(exclude={"provider_id"})))
        match = next((r for r in result["recommendations"] if r["id"] == args.provider_id), None)
        return {"match": match, "fictional": True} if match else {"error": "no_match", "message": "Provider is absent or fails the requested constraints."}
    registry.register(Tool("services.search", "Rank fictional demo providers by category (food, plumber, shopping), coordinates, budget, preferences and availability. Supply remembered constraints; never invent coordinates. All results are fictional and must be labeled as demo data.",
                           DiscoveryRequest, Permission.READ_ONLY, search, cache_ttl=60))
    registry.register(Tool("services.explain_match", "Explain a fictional provider's deterministic score for given search criteria.",
                           ExplainRequest, Permission.READ_ONLY, explain, cache_ttl=60))
