"""Pricing and sizing math for Kalshi binary contracts.

Kalshi contracts settle at $1.00 (win) or $0.00 (loss) and trade in whole cents
between 1 and 99. All prices here are integer cents unless a name says otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MIN_PRICE_CENTS: int = 1
MAX_PRICE_CENTS: int = 99
PAYOUT_DOLLARS: float = 1.00
TAKER_FEE_RATE: float = 0.07


def _validate_price(price_cents: int) -> None:
    if isinstance(price_cents, bool) or not isinstance(price_cents, int):
        raise TypeError(f"price_cents must be int, got {type(price_cents).__name__}")
    if not MIN_PRICE_CENTS <= price_cents <= MAX_PRICE_CENTS:
        raise ValueError(
            f"price_cents must be in [{MIN_PRICE_CENTS}, {MAX_PRICE_CENTS}], got {price_cents}"
        )


def _validate_probability(p: float, name: str = "probability") -> None:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {p}")


def price_to_probability(price_cents: int) -> float:
    """Return the market-implied probability (0.0-1.0) of a contract priced in cents."""
    _validate_price(price_cents)
    return price_cents / 100.0


def probability_to_price(probability: float) -> int:
    """Return the nearest tradable price in cents for a probability, clamped to [1, 99]."""
    _validate_probability(probability)
    return min(MAX_PRICE_CENTS, max(MIN_PRICE_CENTS, round(probability * 100)))


def price_to_decimal_odds(price_cents: int) -> float:
    """Return decimal odds (total return per $1 staked, stake included). 16c -> 6.25."""
    _validate_price(price_cents)
    return 100.0 / price_cents


def price_to_american_odds(price_cents: int) -> float:
    """Return American moneyline odds. 16c -> +525, 80c -> -400, 50c -> +100."""
    _validate_price(price_cents)
    if price_cents <= 50:
        return 100.0 * (100 - price_cents) / price_cents
    return -100.0 * price_cents / (100 - price_cents)


def american_odds_to_probability(american_odds: float) -> float:
    """Return the implied probability (vig included) of American odds."""
    if -100.0 < american_odds < 100.0:
        raise ValueError(f"American odds must be <= -100 or >= +100, got {american_odds}")
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    return -american_odds / (-american_odds + 100.0)


def decimal_odds_to_probability(decimal_odds: float) -> float:
    """Return the implied probability (vig included) of decimal odds."""
    if decimal_odds <= 1.0:
        raise ValueError(f"decimal odds must be > 1.0, got {decimal_odds}")
    return 1.0 / decimal_odds


def taker_fee_dollars(price_cents: int, contracts: int = 1, rate: float = TAKER_FEE_RATE) -> float:
    """Return the Kalshi trading fee in dollars: ceil_to_cent(rate * C * P * (1 - P)).

    `rate` defaults to the standard taker rate; pass the market's schedule rate when it differs.
    """
    _validate_price(price_cents)
    if contracts < 0:
        raise ValueError(f"contracts must be >= 0, got {contracts}")
    p = price_cents / 100.0
    raw_cents = rate * contracts * p * (1.0 - p) * 100.0
    return math.ceil(round(raw_cents, 9)) / 100.0


@dataclass(frozen=True, slots=True)
class KellySizing:
    """Kelly output for a single contract side."""

    cost_per_contract: float  # premium + per-contract fee, dollars
    edge: float  # expected profit per $1 of cost
    full_kelly_fraction: float  # optimal fraction of bankroll, before risk scaling
    applied_fraction: float  # full Kelly * risk_fraction, capped
    stake_dollars: float  # dollars committed (contracts * cost)
    contracts: int  # whole contracts to buy


def kelly_size(
    win_probability: float,
    price_cents: int,
    bankroll: float,
    risk_fraction: float = 0.25,
    fee_per_contract: float = 0.0,
    max_fraction: float = 1.0,
) -> KellySizing:
    """Size a long position in a binary contract that pays $1.00 on a win.

    With all-in cost c (premium + fee) and payout 1, net odds are b = (1 - c) / c and the
    Kelly fraction reduces to f* = (p - c) / (1 - c). The applied fraction is
    f* * risk_fraction, capped at `max_fraction`, and contracts are floored to whole units.

    For a NO position pass the NO price and 1 - p_yes as `win_probability`.
    """
    _validate_probability(win_probability, "win_probability")
    _validate_price(price_cents)
    if bankroll < 0:
        raise ValueError(f"bankroll must be >= 0, got {bankroll}")
    if not 0.0 <= risk_fraction <= 1.0:
        raise ValueError(f"risk_fraction must be in [0, 1], got {risk_fraction}")
    if not 0.0 <= max_fraction <= 1.0:
        raise ValueError(f"max_fraction must be in [0, 1], got {max_fraction}")
    if fee_per_contract < 0:
        raise ValueError(f"fee_per_contract must be >= 0, got {fee_per_contract}")

    cost = price_cents / 100.0 + fee_per_contract
    edge = win_probability * PAYOUT_DOLLARS / cost - 1.0
    if cost >= PAYOUT_DOLLARS:
        return KellySizing(cost, edge, 0.0, 0.0, 0.0, 0)

    full_kelly = max(0.0, (win_probability - cost) / (PAYOUT_DOLLARS - cost))
    applied = min(full_kelly * risk_fraction, max_fraction)
    contracts = math.floor(round(bankroll * applied / cost, 9))
    return KellySizing(
        cost_per_contract=cost,
        edge=edge,
        full_kelly_fraction=full_kelly,
        applied_fraction=applied,
        stake_dollars=contracts * cost,
        contracts=contracts,
    )
