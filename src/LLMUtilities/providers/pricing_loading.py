from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, TypeVar

from pydantic import BaseModel

from ..exceptions import PricingUnavailableError

CATALOGUE_SCHEMA_VERSION = 1

PricingT = TypeVar("PricingT", bound=BaseModel)


def load_pricing_catalogue(path: Path, model_cls: type[PricingT]) -> list[PricingT]:
    """
    Load a schema-v1 pricing catalogue file into a list of provider-owned
    pricing records.

    This is shared *mechanics* only (JSON parsing, schema-version check,
    validation against whichever Pydantic type the caller supplies) — it does
    not know or care what fields a provider's pricing type has.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(raw, dict) or "entries" not in raw:
        raise ValueError(
            f"{path} must be a schema-v1 catalogue object with an 'entries' array."
        )

    schema_version = int(raw.get("schema_version", CATALOGUE_SCHEMA_VERSION))
    if schema_version != CATALOGUE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported pricing schema_version {schema_version} in {path}; "
            f"expected {CATALOGUE_SCHEMA_VERSION}."
        )

    return [model_cls.model_validate(entry) for entry in raw["entries"]]


def _parse_date(value: Optional[str]) -> Optional[date]:
    if value is None:
        return None
    return datetime.fromisoformat(value).date()


def select_pricing(
    entries: Sequence[PricingT],
    model: str,
    *,
    provider_name: str,
    effective_at: Optional[datetime] = None,
) -> PricingT:
    """
    Resolve the pricing record applicable to ``model`` at ``effective_at``
    (defaulting to now) from a provider's own pricing entries.

    Raises ``PricingUnavailableError`` rather than implying the model itself
    is unsupported — a model can be supported with no available pricing.
    """
    at = (effective_at or datetime.now(timezone.utc)).date()

    candidates = [
        entry
        for entry in entries
        if getattr(entry, "canonical_model_id") == model
    ]

    applicable = []
    for entry in candidates:
        effective_from = _parse_date(getattr(entry, "effective_from", None))
        effective_until = _parse_date(getattr(entry, "effective_until", None))

        if effective_from is not None and effective_from > at:
            continue
        if effective_until is not None and effective_until <= at:
            continue
        applicable.append(entry)

    if not applicable:
        raise PricingUnavailableError(
            f"No pricing record available for {provider_name!r} model {model!r} "
            f"effective at {at.isoformat()}."
        )

    applicable.sort(
        key=lambda entry: _parse_date(getattr(entry, "effective_from", None))
        or date.min
    )
    return applicable[-1]


def select_pricing_for_response(
    entries: Sequence[PricingT],
    *,
    resolved_model: str,
    requested_model: Optional[str],
    provider_name: str,
    effective_at: Optional[datetime] = None,
) -> PricingT:
    """
    Resolve pricing for a completed response, which carries both a resolved
    (provider-reported, possibly dated/snapshotted) model and the originally
    requested model.

    A provider may bill a more specific model than it was asked for (§8.1) -
    e.g. a caller requests the canonical alias "claude-sonnet-5" but the SDK
    reports back a dated snapshot id that isn't itself a catalogue entry.
    Pricing is looked up by the resolved model first since that's what was
    actually billed; if that's not a catalogue entry, it falls back to the
    requested model, which is guaranteed to be a catalogue id whenever the
    provider resolved its own default before calling the SDK.
    """
    try:
        return select_pricing(
            entries, resolved_model, provider_name=provider_name, effective_at=effective_at
        )
    except PricingUnavailableError:
        if requested_model and requested_model != resolved_model:
            return select_pricing(
                entries,
                requested_model,
                provider_name=provider_name,
                effective_at=effective_at,
            )
        raise
