import asyncio
import base64
import json
from decimal import Decimal

import httpx
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

from kalshi_client import (
    KalshiAPIError,
    KalshiClient,
    KalshiRateLimitError,
    Orderbook,
    RequestSigner,
    format_count,
    format_price,
    snap_price,
)

BASE = "https://external-api.demo.kalshi.co/trade-api/v2"


@pytest.fixture
def ed_key():
    return ed25519.Ed25519PrivateKey.generate()


def make_client(handler, signer=None, **kwargs):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return KalshiClient(signer, environment="demo", http_client=http, backoff_base=0, **kwargs)


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- signing


def test_ed25519_signature_verifies_and_strips_query(ed_key):
    signer = RequestSigner("key-1", ed_key)
    h = signer.headers("get", "/trade-api/v2/portfolio/orders?limit=5", timestamp_ms=1700000000000)
    assert h["KALSHI-ACCESS-KEY"] == "key-1"
    assert h["KALSHI-ACCESS-TIMESTAMP"] == "1700000000000"
    ed_key.public_key().verify(
        base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"]), b"1700000000000GET/trade-api/v2/portfolio/orders"
    )


def test_rsa_pss_signature_verifies():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    signer = RequestSigner.from_pem("key-2", pem)
    h = signer.headers("POST", "/trade-api/v2/portfolio/events/orders", timestamp_ms=1)
    key.public_key().verify(
        base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"]),
        b"1POST/trade-api/v2/portfolio/events/orders",
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
        hashes.SHA256(),
    )


def test_client_signs_full_api_path(ed_key):
    seen = {}

    def handler(request):
        seen.update(request.headers)
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"balance": 100})

    client = make_client(handler, RequestSigner("k", ed_key))
    run(client.get_balance())
    assert seen["url"] == f"{BASE}/portfolio/balance"
    message = f"{seen['kalshi-access-timestamp']}GET/trade-api/v2/portfolio/balance".encode()
    ed_key.public_key().verify(base64.b64decode(seen["kalshi-access-signature"]), message)


def test_private_endpoint_without_signer_raises():
    client = make_client(lambda r: httpx.Response(200, json={}))
    with pytest.raises(RuntimeError):
        run(client.get_balance())


# ---------------------------------------------------------------- fixed point


def test_format_price_and_count():
    assert format_price("0.56") == "0.5600"
    assert format_price(Decimal("0.0001")) == "0.0001"
    assert format_price(0.16) == "0.1600"
    assert format_count(10) == "10.00"
    assert format_count("2.5") == "2.50"
    for bad in ("0", "1", "1.5", "0.12345"):
        with pytest.raises(ValueError):
            format_price(bad)
    with pytest.raises(TypeError):
        format_price(56)
    for bad in (0, "-1", "1.005"):
        with pytest.raises(ValueError):
            format_count(bad)


def test_snap_price_rounds_away_from_crossing():
    ranges = [
        {"start": "0.0000", "end": "0.1000", "step": "0.0010"},
        {"start": "0.1000", "end": "0.9000", "step": "0.0100"},
    ]
    assert snap_price("0.5678", ranges, "bid") == Decimal("0.56")
    assert snap_price("0.5678", ranges, "ask") == Decimal("0.57")
    assert snap_price("0.0525", ranges, "bid") == Decimal("0.052")
    with pytest.raises(ValueError):
        snap_price("0.95", ranges, "bid")


def test_orderbook_implied_asks():
    book = Orderbook.from_response(
        {"orderbook_fp": {"yes_dollars": [["0.4000", "5.00"], ["0.4200", "1.00"]], "no_dollars": [["0.5500", "3.00"]]}}
    )
    assert book.best_yes_bid == Decimal("0.42")
    assert book.best_yes_ask == Decimal("0.45")
    assert book.best_no_ask == Decimal("0.58")
    empty = Orderbook.from_response({"orderbook_fp": {"yes_dollars": [], "no_dollars": None}})
    assert empty.best_yes_bid is None and empty.best_yes_ask is None


# ---------------------------------------------------------------- orders


def test_create_order_payload(ed_key):
    seen = {}

    def handler(request):
        seen["method"], seen["url"] = request.method, str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"order_id": "o1", "fill_count": "0.00", "remaining_count": "10.00", "ts_ms": 1})

    client = make_client(handler, RequestSigner("k", ed_key))
    resp = run(client.create_order("HIGHNY-24JAN01-T60", "bid", 10, "0.56", post_only=True))
    assert resp["order_id"] == "o1"
    assert seen["method"] == "POST"
    assert seen["url"] == f"{BASE}/portfolio/events/orders"
    body = seen["body"]
    assert body["count"] == "10.00" and body["price"] == "0.5600"
    assert body["time_in_force"] == "good_till_canceled"
    assert body["self_trade_prevention_type"] == "taker_at_cross"
    assert body["post_only"] is True and "reduce_only" not in body
    assert len(body["client_order_id"]) == 36


def test_create_order_validation(ed_key):
    client = make_client(lambda r: httpx.Response(201, json={}), RequestSigner("k", ed_key))
    with pytest.raises(ValueError):
        run(client.create_order("T", "buy", 1, "0.5"))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        run(client.create_order("T", "bid", 1, "0.5", reduce_only=True))
    with pytest.raises(ValueError):
        run(client.create_order("T", "bid", 1, "0.5", time_in_force="immediate_or_cancel", expiration_time=1))


def test_cancel_order_requires_routing(ed_key):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"order_id": "o1", "reduced_by": "1.00", "ts_ms": 1})

    client = make_client(handler, RequestSigner("k", ed_key))
    with pytest.raises(ValueError):
        run(client.cancel_order("o1"))
    run(client.cancel_order("o1", market_ticker="ABC"))
    assert seen["url"] == f"{BASE}/portfolio/events/orders/o1?market_ticker=ABC"


# ---------------------------------------------------------------- retries & errors


def test_429_is_retried_for_post(ed_key):
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, json={"error": "too many requests"})
        return httpx.Response(201, json={"order_id": "o1"})

    client = make_client(handler, RequestSigner("k", ed_key))
    assert run(client.create_order("T", "ask", 1, "0.5"))["order_id"] == "o1"
    assert len(calls) == 3


def test_post_5xx_is_not_retried(ed_key):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(500, json={"code": "internal", "message": "boom"})

    client = make_client(handler, RequestSigner("k", ed_key))
    with pytest.raises(KalshiAPIError) as exc:
        run(client.create_order("T", "bid", 1, "0.5"))
    assert exc.value.status == 500 and exc.value.code == "internal"
    assert len(calls) == 1


def test_get_5xx_retried_then_raises():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503, text="unavailable")

    client = make_client(handler, max_retries=2)
    with pytest.raises(KalshiAPIError):
        run(client.get_market("ABC"))
    assert len(calls) == 3


def test_persistent_429_raises_rate_limit_error():
    client = make_client(lambda r: httpx.Response(429, json={"error": "too many requests"}), max_retries=1)
    with pytest.raises(KalshiRateLimitError) as exc:
        run(client.get_markets())
    assert exc.value.message == "too many requests"


# ---------------------------------------------------------------- pagination


def test_iter_markets_follows_cursor():
    pages = {
        None: {"markets": [{"ticker": "A"}, {"ticker": "B"}], "cursor": "c1"},
        "c1": {"markets": [{"ticker": "C"}], "cursor": ""},
    }
    seen_params = []

    def handler(request):
        seen_params.append(dict(request.url.params))
        return httpx.Response(200, json=pages[request.url.params.get("cursor")])

    async def collect():
        client = make_client(handler)
        return [m["ticker"] async for m in client.iter_markets(status="open", limit=2)]

    assert run(collect()) == ["A", "B", "C"]
    assert seen_params[0] == {"status": "open", "limit": "2"}
    assert seen_params[1]["cursor"] == "c1"
