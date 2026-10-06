"""API cost of a run from its recorded token usage.

Prices are USD per million tokens (first-party Claude API, as of 2026-09).
Cache writes are priced at the 5-minute rate; the pipeline's 1-hour system
prompt cache writes cost more, so this slightly undercounts them. A batch
run is half price, except a paused item's synchronous continuation, which
the recorded usage doesn't separate — so a batch figure is a lower bound.
"""
from __future__ import annotations

PRICES = {
    # model: (input, output, cache write, cache read)
    "claude-sonnet-5-5": (2.00, 10.00, 2.50, 0.20),
    "claude-sonnet-5": (2.00, 10.00, 2.50, 0.20),
    "claude-opus-4-8": (5.00, 25.00, 6.25, 0.50),
}
BATCH_DISCOUNT = 0.5


def usd(usage: dict, model: str, *, via_batch: bool = False) -> float | None:
    """Estimated cost of `usage` (the four token counts); None for an unpriced model."""
    prices = PRICES.get(model)
    if prices is None:
        return None
    tokens = [usage.get(k) or 0 for k in
              ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")]
    total = sum(t * p for t, p in zip(tokens, prices)) / 1e6
    return round(total * (BATCH_DISCOUNT if via_batch else 1.0), 3)
