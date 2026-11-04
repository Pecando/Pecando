"""Async REST client for the Kalshi Trade API v2.

Prices are fixed-point dollar strings (up to 4 decimals, e.g. "0.5600") and counts are
fixed-point strings (up to 2 decimals, e.g. "10.00"). This module accepts `Decimal`,
`str` or `float` for both and never rounds silently: values with too much precision are
rejected. Use `snap_price` with a market's `price_ranges` to land on its tick grid.
"""

from __future__ import annotations

import asyncio
import base64
import os
import random
import time
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

Environment = Literal["prod", "demo"]
OrderSide = Literal["bid", "ask"]
TimeInForce = Literal["fill_or_kill", "good_till_canceled", "immediate_or_cancel"]
SelfTradePrevention = Literal["taker_at_cross", "maker"]
Number = Decimal | str | float

BASE_URLS: dict[str, str] = {
    "prod": "https://external-api.kalshi.com/trade-api/v2",
    "demo": "https://external-api.demo.kalshi.co/trade-api/v2",
}

PRICE_QUANTUM = Decimal("0.0001")
COUNT_QUANTUM = Decimal("0.01")
_RETRYABLE_STATUS = frozenset({500, 502, 503, 504})


# --------------------------------------------------------------------------- errors


class KalshiAPIError(Exception):
    """Non-2xx response from Kalshi."""

    def __init__(self, status: int, code: str | None, message: str, details: str | None, body: Any):
        super().__init__(f"HTTP {status} {code or ''}: {message}".strip())
        self.status = status
        self.code = code
        self.message = message
        self.details = details
        self.body = body


class KalshiRateLimitError(KalshiAPIError):
    """HTTP 429 that persisted through every retry."""


def _api_error(response: httpx.Response) -> KalshiAPIError:
    try:
        body: Any = response.json()
    except ValueError:
        body = response.text
    err = body.get("error", body) if isinstance(body, dict) else body
    if isinstance(err, dict):
        code, message, details = err.get("code"), err.get("message", ""), err.get("details")
    else:
        code, message, details = None, str(err), None
    cls = KalshiRateLimitError if response.status_code == 429 else KalshiAPIError
    return cls(response.status_code, code, message, details, body)


# --------------------------------------------------------------------------- signing


class RequestSigner:
    """Builds Kalshi auth headers. Signs `timestamp_ms + METHOD + path` (query stripped).

    Ed25519 keys sign the message directly; RSA keys use PSS/SHA-256 with a 32-byte salt.
    """

    def __init__(self, key_id: str, private_key: Ed25519PrivateKey | RSAPrivateKey):
        if not isinstance(private_key, (Ed25519PrivateKey, RSAPrivateKey)):
            raise TypeError("private_key must be an Ed25519 or RSA private key")
        self.key_id = key_id
        self._key = private_key

    @classmethod
    def from_pem(cls, key_id: str, pem: bytes | str, password: bytes | None = None) -> RequestSigner:
        """Load a PEM-encoded private key."""
        data = pem.encode() if isinstance(pem, str) else pem
        key = serialization.load_pem_private_key(data, password=password)
        return cls(key_id, key)  # type: ignore[arg-type]

    @classmethod
    def from_file(cls, key_id: str, path: str | Path, password: bytes | None = None) -> RequestSigner:
        """Load a PEM private key from disk."""
        return cls.from_pem(key_id, Path(path).read_bytes(), password)

    def sign(self, message: bytes) -> bytes:
        """Return the raw signature over `message`."""
        if isinstance(self._key, Ed25519PrivateKey):
            return self._key.sign(message)
        return self._key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )

    def headers(self, method: str, path: str, timestamp_ms: int | None = None) -> dict[str, str]:
        """Return the three KALSHI-ACCESS-* headers for a request to the absolute URL path."""
        ts = str(timestamp_ms if timestamp_ms is not None else time.time_ns() // 1_000_000)
        signed_path = path.split("?", 1)[0]
        signature = self.sign(f"{ts}{method.upper()}{signed_path}".encode())
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
        }


# --------------------------------------------------------------------------- fixed-point helpers


def _to_decimal(value: Number, name: str) -> Decimal:
    if isinstance(value, bool) or isinstance(value, int):
        raise TypeError(f"{name} must be Decimal, str or float (int is ambiguous: cents or dollars?)")
    try:
        return Decimal(str(value)) if isinstance(value, float) else Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{name} is not a number: {value!r}") from exc


def format_price(price: Number) -> str:
    """Return a price as a 4-decimal dollar string. Must be in (0, 1) with <= 4 decimals."""
    p = _to_decimal(price, "price")
    if not Decimal(0) < p < Decimal(1):
        raise ValueError(f"price must be strictly between 0 and 1 dollars, got {p}")
    if p != p.quantize(PRICE_QUANTUM):
        raise ValueError(f"price has more than 4 decimal places: {p}")
    return str(p.quantize(PRICE_QUANTUM))


def format_count(count: Number | int) -> str:
    """Return a contract count as a 2-decimal string. Must be > 0 with <= 2 decimals."""
    c = Decimal(count) if isinstance(count, int) and not isinstance(count, bool) else _to_decimal(count, "count")
    if c <= 0:
        raise ValueError(f"count must be > 0, got {c}")
    if c != c.quantize(COUNT_QUANTUM):
        raise ValueError(f"count has more than 2 decimal places: {c}")
    return str(c.quantize(COUNT_QUANTUM))


def snap_price(price: Number, price_ranges: Sequence[Mapping[str, str]], side: OrderSide) -> Decimal:
    """Snap a price onto a market's tick grid, never crossing to a worse price.

    Bids round down and asks round up. `price_ranges` is the market's list of
    {"start", "end", "step"} dollar strings.
    """
    p = _to_decimal(price, "price")
    rounding = ROUND_DOWN if side == "bid" else ROUND_UP
    for band in price_ranges:
        start, end, step = Decimal(band["start"]), Decimal(band["end"]), Decimal(band["step"])
        if start <= p <= end:
            ticks = ((p - start) / step).to_integral_value(rounding=rounding)
            return start + ticks * step
    raise ValueError(f"price {p} is outside every price range")


@dataclass(frozen=True, slots=True)
class Orderbook:
    """Resting bids on each side. Levels are (price, size), best first.

    Kalshi books show bids only: a YES ask is implied by a NO bid at 1 - price.
    """

    yes_bids: tuple[tuple[Decimal, Decimal], ...]
    no_bids: tuple[tuple[Decimal, Decimal], ...]

    @classmethod
    def from_response(cls, body: Mapping[str, Any]) -> Orderbook:
        book = body["orderbook_fp"]

        def levels(raw: Sequence[Sequence[str]] | None) -> tuple[tuple[Decimal, Decimal], ...]:
            parsed = ((Decimal(px), Decimal(qty)) for px, qty in raw or ())
            return tuple(sorted(parsed, key=lambda lvl: lvl[0], reverse=True))

        return cls(levels(book.get("yes_dollars")), levels(book.get("no_dollars")))

    @property
    def best_yes_bid(self) -> Decimal | None:
        return self.yes_bids[0][0] if self.yes_bids else None

    @property
    def best_yes_ask(self) -> Decimal | None:
        return Decimal(1) - self.no_bids[0][0] if self.no_bids else None

    @property
    def best_no_bid(self) -> Decimal | None:
        return self.no_bids[0][0] if self.no_bids else None

    @property
    def best_no_ask(self) -> Decimal | None:
        return Decimal(1) - self.yes_bids[0][0] if self.yes_bids else None


# --------------------------------------------------------------------------- client


def _clean(params: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if v is not None}


class KalshiClient:
    """Async Kalshi REST client.

    Retries: 429 is retried for every method (the request was not processed). 5xx and
    transport errors are retried for GET only; order writes are never resent after an
    ambiguous failure. Reconcile with `get_orders` using the order's `client_order_id`.
    """

    def __init__(
        self,
        signer: RequestSigner | None = None,
        *,
        environment: Environment = "demo",
        base_url: str | None = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        backoff_base: float = 0.05,
        backoff_cap: float = 2.0,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.base_url = (base_url or BASE_URLS[environment]).rstrip("/")
        self._base_path = urlsplit(self.base_url).path
        self._signer = signer
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        self._owns_http = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=timeout)

    @classmethod
    def from_env(cls, environment: Environment | None = None, **kwargs: Any) -> KalshiClient:
        """Build from KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH and optional KALSHI_ENV."""
        signer = RequestSigner.from_file(os.environ["KALSHI_API_KEY_ID"], os.environ["KALSHI_PRIVATE_KEY_PATH"])
        env = environment or os.environ.get("KALSHI_ENV", "demo")
        if env not in BASE_URLS:
            raise ValueError(f"KALSHI_ENV must be one of {sorted(BASE_URLS)}, got {env!r}")
        return cls(signer, environment=env, **kwargs)  # type: ignore[arg-type]

    async def __aenter__(self) -> KalshiClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying HTTP client if this instance created it."""
        if self._owns_http:
            await self._http.aclose()

    # ------------------------------------------------------------------ transport

    def _backoff(self, attempt: int) -> float:
        return random.uniform(0, min(self._backoff_cap, self._backoff_base * 2**attempt))

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Mapping[str, Any] | None = None,
        auth: bool | None = None,
    ) -> Any:
        """Send a request. `auth=None` signs when a signer is set; `True` requires one."""
        method = method.upper()
        url = f"{self.base_url}{path}"
        safe = method == "GET"
        for attempt in range(self._max_retries + 1):
            headers: dict[str, str] = {}
            if self._signer is not None and auth is not False:
                headers = self._signer.headers(method, self._base_path + path)
            elif auth:
                raise RuntimeError(f"{method} {path} requires a RequestSigner")
            try:
                response = await self._http.request(
                    method, url, params=_clean(params or {}), json=json, headers=headers
                )
            except httpx.TransportError:
                if safe and attempt < self._max_retries:
                    await asyncio.sleep(self._backoff(attempt))
                    continue
                raise
            retryable = response.status_code == 429 or (safe and response.status_code in _RETRYABLE_STATUS)
            if retryable and attempt < self._max_retries:
                await asyncio.sleep(self._backoff(attempt))
                continue
            if response.is_error:
                raise _api_error(response)
            return response.json() if response.content else None
        raise AssertionError("unreachable")

    async def iter_pages(self, path: str, params: Mapping[str, Any] | None = None) -> AsyncIterator[dict[str, Any]]:
        """Yield every page of a cursor-paginated GET endpoint."""
        query = dict(params or {})
        while True:
            page = await self._request("GET", path, params=query, auth=path.startswith("/portfolio") or None)
            yield page
            cursor = page.get("cursor")
            if not cursor:
                return
            query["cursor"] = cursor

    async def _iter_items(self, path: str, key: str, params: Mapping[str, Any]) -> AsyncIterator[dict[str, Any]]:
        async for page in self.iter_pages(path, params):
            for item in page.get(key, []):
                yield item

    # ------------------------------------------------------------------ exchange & markets

    async def get_exchange_status(self) -> dict[str, Any]:
        """Return exchange and trading-active flags."""
        return await self._request("GET", "/exchange/status")

    async def get_markets(self, **filters: Any) -> dict[str, Any]:
        """Return one page of markets (`markets`, `cursor`). Filters map to query params."""
        return await self._request("GET", "/markets", params=filters)

    def iter_markets(self, **filters: Any) -> AsyncIterator[dict[str, Any]]:
        """Yield every market matching the filters across all pages."""
        return self._iter_items("/markets", "markets", filters)

    async def get_market(self, ticker: str) -> dict[str, Any]:
        """Return a single market object."""
        body = await self._request("GET", f"/markets/{ticker}")
        return body["market"]

    async def get_orderbook(self, ticker: str, depth: int = 0) -> Orderbook:
        """Return the parsed orderbook. `depth` 0 means all levels, 1-100 limits each side."""
        body = await self._request(
            "GET", f"/markets/{ticker}/orderbook", params={"depth": depth}
        )
        return Orderbook.from_response(body)

    async def get_event(self, event_ticker: str, **params: Any) -> dict[str, Any]:
        """Return an event (`event`, and `markets` when requested)."""
        return await self._request("GET", f"/events/{event_ticker}", params=params)

    # ------------------------------------------------------------------ portfolio

    async def get_balance(self, subaccount: int | None = None, exchange_index: int | None = None) -> dict[str, Any]:
        """Return balance fields. Prefer the fills websocket over polling this endpoint."""
        return await self._request(
            "GET", "/portfolio/balance", params={"subaccount": subaccount, "exchange_index": exchange_index}, auth=True
        )

    async def get_positions(self, **filters: Any) -> dict[str, Any]:
        """Return one page of positions (`market_positions`, `event_positions`, `cursor`)."""
        return await self._request("GET", "/portfolio/positions", params=filters, auth=True)

    async def get_orders(self, **filters: Any) -> dict[str, Any]:
        """Return one page of orders (`orders`, `cursor`)."""
        return await self._request("GET", "/portfolio/orders", params=filters, auth=True)

    def iter_orders(self, **filters: Any) -> AsyncIterator[dict[str, Any]]:
        """Yield every order matching the filters across all pages."""
        return self._iter_items("/portfolio/orders", "orders", filters)

    async def get_order(self, order_id: str) -> dict[str, Any]:
        """Return a single order object."""
        body = await self._request("GET", f"/portfolio/orders/{order_id}", auth=True)
        return body["order"]

    async def get_fills(self, **filters: Any) -> dict[str, Any]:
        """Return one page of fills (`fills`, `cursor`)."""
        return await self._request("GET", "/portfolio/fills", params=filters, auth=True)

    def iter_fills(self, **filters: Any) -> AsyncIterator[dict[str, Any]]:
        """Yield every fill matching the filters across all pages."""
        return self._iter_items("/portfolio/fills", "fills", filters)

    # ------------------------------------------------------------------ orders

    async def create_order(
        self,
        ticker: str,
        side: OrderSide,
        count: Number | int,
        price: Number,
        *,
        time_in_force: TimeInForce = "good_till_canceled",
        self_trade_prevention_type: SelfTradePrevention = "taker_at_cross",
        client_order_id: str | None = None,
        post_only: bool | None = None,
        reduce_only: bool | None = None,
        expiration_time: int | None = None,
        cancel_order_on_pause: bool | None = None,
        subaccount: int | None = None,
        order_group_id: str | None = None,
        exchange_index: int | None = None,
    ) -> dict[str, Any]:
        """Place a limit order on the YES leg: `bid` buys YES, `ask` sells YES.

        To buy NO at price q, send an `ask` at 1 - q. `price` is in dollars. A UUID
        `client_order_id` is generated when omitted and echoed in the response.
        """
        if side not in ("bid", "ask"):
            raise ValueError(f"side must be 'bid' or 'ask', got {side!r}")
        if reduce_only and time_in_force != "immediate_or_cancel":
            raise ValueError("reduce_only requires time_in_force='immediate_or_cancel'")
        if expiration_time is not None and time_in_force == "immediate_or_cancel":
            raise ValueError("expiration_time cannot be combined with immediate_or_cancel")
        payload = _clean(
            {
                "ticker": ticker,
                "side": side,
                "count": format_count(count),
                "price": format_price(price),
                "time_in_force": time_in_force,
                "self_trade_prevention_type": self_trade_prevention_type,
                "client_order_id": client_order_id or str(uuid.uuid4()),
                "post_only": post_only,
                "reduce_only": reduce_only,
                "expiration_time": expiration_time,
                "cancel_order_on_pause": cancel_order_on_pause,
                "subaccount": subaccount,
                "order_group_id": order_group_id,
                "exchange_index": exchange_index,
            }
        )
        return await self._request("POST", "/portfolio/events/orders", json=payload, auth=True)

    async def cancel_order(
        self,
        order_id: str,
        *,
        market_ticker: str | None = None,
        exchange_index: int | None = None,
        subaccount: int | None = None,
    ) -> dict[str, Any]:
        """Cancel a resting order. Pass `market_ticker` or `exchange_index` so it can be routed."""
        if market_ticker is None and exchange_index is None:
            raise ValueError("cancel_order needs market_ticker or exchange_index to route to the right shard")
        return await self._request(
            "DELETE",
            f"/portfolio/events/orders/{order_id}",
            params={"market_ticker": market_ticker, "exchange_index": exchange_index, "subaccount": subaccount},
            auth=True,
        )

    async def cancel_all_orders(self, subaccount: int | None = None) -> None:
        """Cancel every resting order (all subaccounts when `subaccount` is omitted)."""
        await self._request("DELETE", "/portfolio/events/orders", params={"subaccount": subaccount}, auth=True)
