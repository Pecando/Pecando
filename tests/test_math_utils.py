import pytest

from math_utils import (
    american_odds_to_probability,
    decimal_odds_to_probability,
    kelly_size,
    price_to_american_odds,
    price_to_decimal_odds,
    price_to_probability,
    probability_to_price,
    taker_fee_dollars,
)


def test_probability_conversion():
    assert price_to_probability(16) == pytest.approx(0.16)
    assert probability_to_price(0.004) == 1
    assert probability_to_price(0.999) == 99
    with pytest.raises(ValueError):
        price_to_probability(0)
    with pytest.raises(ValueError):
        price_to_probability(100)
    with pytest.raises(TypeError):
        price_to_probability(0.16)


@pytest.mark.parametrize(
    ("cents", "american", "decimal"),
    [(16, 525.0, 6.25), (50, 100.0, 2.0), (80, -400.0, 1.25), (1, 9900.0, 100.0), (99, -9900.0, 100 / 99)],
)
def test_odds(cents, american, decimal):
    assert price_to_american_odds(cents) == pytest.approx(american)
    assert price_to_decimal_odds(cents) == pytest.approx(decimal)
    assert american_odds_to_probability(price_to_american_odds(cents)) == pytest.approx(cents / 100)
    assert decimal_odds_to_probability(price_to_decimal_odds(cents)) == pytest.approx(cents / 100)


def test_taker_fee_rounds_up_to_cent():
    # 0.07 * 100 * 0.5 * 0.5 = 1.75 exactly
    assert taker_fee_dollars(50, 100) == pytest.approx(1.75)
    # 0.07 * 1 * 0.5 * 0.5 = 0.0175 -> 0.02
    assert taker_fee_dollars(50, 1) == pytest.approx(0.02)
    assert taker_fee_dollars(50, 0) == 0.0


def test_kelly_matches_closed_form():
    # p=0.30, c=0.16 -> f* = 0.14 / 0.84
    r = kelly_size(0.30, 16, bankroll=1000.0, risk_fraction=1.0)
    assert r.full_kelly_fraction == pytest.approx(0.14 / 0.84)
    assert r.edge == pytest.approx(0.30 / 0.16 - 1)
    assert r.contracts == int(1000 * (0.14 / 0.84) / 0.16)
    assert r.stake_dollars <= 1000 * r.applied_fraction + 1e-9


def test_kelly_risk_fraction_and_cap():
    full = kelly_size(0.30, 16, 1000.0, risk_fraction=1.0)
    quarter = kelly_size(0.30, 16, 1000.0, risk_fraction=0.25)
    assert quarter.applied_fraction == pytest.approx(full.full_kelly_fraction * 0.25)
    capped = kelly_size(0.99, 10, 1000.0, risk_fraction=1.0, max_fraction=0.05)
    assert capped.applied_fraction == pytest.approx(0.05)


def test_kelly_no_edge_and_fees():
    assert kelly_size(0.16, 16, 1000.0).contracts == 0
    assert kelly_size(0.10, 16, 1000.0).full_kelly_fraction == 0.0
    # Edge that a fee wipes out
    assert kelly_size(0.17, 16, 1000.0, fee_per_contract=0.02).contracts == 0
    assert kelly_size(0.995, 99, 1000.0, fee_per_contract=0.01).contracts == 0
