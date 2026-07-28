"""Pydantic models for the pricing module."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class PricingTier(BaseModel):
    """A pricing tier for a model family."""

    name: str
    price_per_input_token_cny: float
    price_per_output_token_cny: float
    currency: Literal["CNY"] = "CNY"


class PricingInfo(BaseModel):
    """Pricing info for a specific model."""

    model: str
    tier: str
    price_per_input_token_cny: float
    price_per_output_token_cny: float
    currency: Literal["CNY"] = "CNY"