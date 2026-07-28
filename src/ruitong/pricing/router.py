"""FastAPI router for pricing endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..config import BridgeConfig
from .models import PricingInfo

router = APIRouter(prefix="/v1", tags=["pricing"])


def _get_pricing_config(request: Request) -> dict[str, dict]:
    """Return pricing config from app state, or fall back to env."""
    config: dict[str, dict] | None = getattr(
        request.app.state, "pricing_config", None
    )
    if config is None:
        config = BridgeConfig.from_env().pricing_config
    return config


@router.get("/pricing")
async def list_pricing(request: Request) -> list[dict]:
    """Return configured pricing for all models."""
    pricing = _get_pricing_config(request)
    if not pricing:
        return []
    results: list[PricingInfo] = []
    for model, tier_data in pricing.items():
        results.append(
            PricingInfo(
                model=model,
                tier=tier_data.get("name", "default"),
                price_per_input_token_cny=float(
                    tier_data.get("price_per_input_token_cny", 0)
                ),
                price_per_output_token_cny=float(
                    tier_data.get("price_per_output_token_cny", 0)
                ),
            )
        )
    return [p.model_dump() for p in results]


@router.get("/pricing/{model}")
async def get_pricing(request: Request, model: str) -> dict:
    """Return pricing for a specific model, or 404."""
    pricing = _get_pricing_config(request)
    if model not in pricing:
        raise HTTPException(status_code=404, detail=f"Model {model!r} not found in pricing config")
    tier_data = pricing[model]
    info = PricingInfo(
        model=model,
        tier=tier_data.get("name", "default"),
        price_per_input_token_cny=float(tier_data.get("price_per_input_token_cny", 0)),
        price_per_output_token_cny=float(tier_data.get("price_per_output_token_cny", 0)),
    )
    return info.model_dump()