from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class CostSummary(BaseModel):
    """
    Common cost projection.

    Not a universal detailed cost model — each provider decides how its
    detailed charges roll into these categories (see each provider's
    ``pricing.py`` and detailed cost types for the breakdown).
    """

    model_config = ConfigDict(extra="forbid")

    input_cost: float
    output_cost: float
    other_cost: float
    total_cost: float
    currency: str
    provider: str
    requested_model: str
    resolved_model: str
