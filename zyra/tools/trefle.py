import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


TREFLE_API_BASE = "https://trefle.io"


@dataclass
class TreflePlantResult:
    common_name: str | None
    scientific_name: str
    slug: str
    image_url: str | None
    growth: dict[str, Any]
    specifications: dict[str, Any]
    distribution: dict[str, Any]
    family: str | None
    genus: str | None
    raw: dict[str, Any]


def _get_token() -> str:
    token = os.getenv("TREFLE_TOKEN", "").strip()
    if not token:
        raise ValueError("TREFLE_TOKEN is not set")
    return token


def _api_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    query = {"token": _get_token()}
    if params:
        query.update({key: value for key, value in params.items() if value is not None})

    url = f"{TREFLE_API_BASE}{path}?{urllib.parse.urlencode(query)}"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Trefle API error ({exc.code}): {body}") from exc


def search_plants(query: str) -> list[dict[str, Any]]:
    payload = _api_get("/api/v1/plants/search", {"q": query})
    return payload.get("data", [])


def get_species(slug_or_id: str) -> dict[str, Any]:
    payload = _api_get(f"/api/v1/species/{slug_or_id}")
    return payload.get("data", {})


def _pick_reference_image(species: dict[str, Any]) -> str | None:
    if species.get("image_url"):
        return species["image_url"]

    images = species.get("images") or {}
    for category in ("habit", "leaf", "flower", "fruit", "bark", "other"):
        entries = images.get(category) or []
        for entry in entries:
            if entry.get("image_url"):
                return entry["image_url"]

    return None


def _best_search_match(query: str, results: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not results:
        return None

    query_lower = query.strip().lower()

    for result in results:
        scientific = (result.get("scientific_name") or "").lower()
        common = (result.get("common_name") or "").lower()
        if query_lower in {scientific, common}:
            return result

    for result in results:
        scientific = (result.get("scientific_name") or "").lower()
        common = (result.get("common_name") or "").lower()
        if query_lower in scientific or scientific in query_lower:
            return result
        if common and (query_lower in common or common in query_lower):
            return result

    return results[0]


def lookup_plant(query: str) -> TreflePlantResult | None:
    if not query or not query.strip():
        return None

    results = search_plants(query.strip())
    match = _best_search_match(query, results)
    if not match:
        return None

    slug = match.get("slug")
    if not slug:
        return None

    species = get_species(slug)
    if not species:
        return None

    return TreflePlantResult(
        common_name=species.get("common_name") or match.get("common_name"),
        scientific_name=species.get("scientific_name") or match.get("scientific_name", ""),
        slug=slug,
        image_url=_pick_reference_image(species),
        growth=species.get("growth") or {},
        specifications=species.get("specifications") or {},
        distribution=species.get("distribution") or {},
        family=species.get("family"),
        genus=species.get("genus"),
        raw=species,
    )


def summarize_for_llm(result: TreflePlantResult) -> dict[str, Any]:
    growth = result.growth or {}
    specs = result.specifications or {}

    return {
        "common_name": result.common_name,
        "scientific_name": result.scientific_name,
        "family": result.family,
        "genus": result.genus,
        "growth_description": growth.get("description"),
        "light": growth.get("light"),
        "soil_humidity": growth.get("soil_humidity"),
        "soil_texture": growth.get("soil_texture"),
        "soil_nutriments": growth.get("soil_nutriments"),
        "minimum_temperature_c": (growth.get("minimum_temperature") or {}).get("deg_c"),
        "maximum_temperature_c": (growth.get("maximum_temperature") or {}).get("deg_c"),
        "ph_minimum": growth.get("ph_minimum"),
        "ph_maximum": growth.get("ph_maximum"),
        "growth_months": growth.get("growth_months"),
        "bloom_months": growth.get("bloom_months"),
        "growth_habit": specs.get("growth_habit"),
        "growth_rate": specs.get("growth_rate"),
        "toxicity": specs.get("toxicity"),
        "native_distribution": (result.distribution or {}).get("native"),
    }
