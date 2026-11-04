"""Async client for the Kalshi Trade API v2: REST via httpx, live market data via websockets.

Credentials come from the environment (see `RequestSigner.from_env`):
    KALSHI_API_KEY_ID        API key ID
    KALSHI_PRIVATE_KEY_PATH  path to the PEM private key (or KALSHI_PRIVATE_KEY with the PEM itself)
    KALSHI_ENV               "demo" (default) or "prod"

Prices are fixed-point dollar strings (up to 4 decimals, e.g. "0.5600") and counts are
fixed-point strings (up to 2 decimals, e.g. "10.00"). This module accepts `Decimal`,
`str` or `float` for both and never rounds silently: values with too much precision are
rejected. Use `snap_price` with a market's `price_ranges` to land on its tick grid.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
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
from websockets.asyncio.client import ClientConnection
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from math_utils import mid_implied_probability

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

    @classmethod
    def from_env(cls) -> RequestSigner:
        """Load from KALSHI_API_KEY_ID plus KALSHI_PRIVATE_KEY (PEM text) or KALSHI_PRIVATE_KEY_PATH."""
        key_id = os.environ.get("KALSHI_API_KEY_ID")
        if not key_id:
            raise RuntimeError("KALSHI_API_KEY_ID is not set")
        pem = os.environ.get("KALSHI_PRIVATE_KEY")
        if pem:
            return cls.from_pem(key_id, pem.replace("\\n", "\n"))
        path = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
        if not path:
            raise RuntimeError("set KALSHI_PRIVATE_KEY_PATH or KALSHI_PRIVATE_KEY")
        return cls.from_file(key_id, path)

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


_TICKS_PER_DOLLAR = 10_000  # 1 tick = $0.0001, the finest Kalshi price increment
_COUNT_SCALE = 100  # counts carry 2 decimals


def _parse_fixed(text: str, places: int) -> int:
    """Parse a fixed-point decimal string into an integer scaled by 10**places, exactly."""
    neg = text.startswith("-")
    whole, _, frac = (text[1:] if neg else text).partition(".")
    if len(frac) > places:
        if frac[places:].strip("0"):
            raise ValueError(f"{text!r} has more than {places} decimal places")
        frac = frac[:places]
    value = int(whole or "0") * 10**places + int(frac.ljust(places, "0"))
    return -value if neg else value


def _ticks_to_decimal(ticks: int) -> Decimal:
    return Decimal(ticks).scaleb(-4)


class Orderbook:
    """Mutable L2 book for one market, stored as integer ticks for cheap updates.

    Kalshi books carry bids only: a YES ask is implied by a NO bid at 1 - price, and a
    NO ask by a YES bid. The best bid on each side is cached so a delta is O(1) unless it
    removes the top level.
    """

    __slots__ = ("ticker", "_yes", "_no", "_best_yes", "_best_no")

    def __init__(self, ticker: str = ""):
        self.ticker = ticker
        self._yes: dict[int, int] = {}
        self._no: dict[int, int] = {}
        self._best_yes: int | None = None
        self._best_no: int | None = None

    @classmethod
    def from_response(cls, body: Mapping[str, Any], ticker: str = "") -> Orderbook:
        """Build from a REST `GET /markets/{ticker}/orderbook` response."""
        book = cls(ticker)
        raw = body["orderbook_fp"]
        book.load(raw.get("yes_dollars"), raw.get("no_dollars"))
        return book

    def load(self, yes_levels: Sequence[Sequence[str]] | None, no_levels: Sequence[Sequence[str]] | None) -> None:
        """Replace the book with full snapshot levels of (dollars, count) strings."""
        self._yes = {_parse_fixed(px, 4): _parse_fixed(qty, 2) for px, qty in yes_levels or ()}
        self._no = {_parse_fixed(px, 4): _parse_fixed(qty, 2) for px, qty in no_levels or ()}
        self._best_yes = max(self._yes, default=None)
        self._best_no = max(self._no, default=None)

    def apply_delta(self, side: str, price_ticks: int, delta: int) -> bool:
        """Apply a size change at one level. Return True if either top of book moved or resized."""
        if side == "yes":
            levels, before = self._yes, self._best_yes
        else:
            levels, before = self._no, self._best_no
        best = before
        qty = levels.get(price_ticks, 0) + delta
        if qty > 0:
            levels[price_ticks] = qty
            if best is None or price_ticks > best:
                best = price_ticks
        else:
            levels.pop(price_ticks, None)
            if price_ticks == best:
                best = max(levels, default=None)
        if side == "yes":
            self._best_yes = best
        else:
            self._best_no = best
        return best != before or price_ticks == best

    # Integer accessors for hot paths: prices in ticks of $0.0001.

    @property
    def best_yes_bid_ticks(self) -> int | None:
        return self._best_yes

    @property
    def best_yes_ask_ticks(self) -> int | None:
        return None if self._best_no is None else _TICKS_PER_DOLLAR - self._best_no

    def size_at(self, side: str, price_ticks: int) -> Decimal:
        """Return resting contracts at a level of the YES or NO bid book."""
        qty = (self._yes if side == "yes" else self._no).get(price_ticks, 0)
        return Decimal(qty).scaleb(-2)

    # Decimal accessors.

    @property
    def yes_bids(self) -> tuple[tuple[Decimal, Decimal], ...]:
        return tuple((_ticks_to_decimal(p), Decimal(q).scaleb(-2)) for p, q in sorted(self._yes.items(), reverse=True))

    @property
    def no_bids(self) -> tuple[tuple[Decimal, Decimal], ...]:
        return tuple((_ticks_to_decimal(p), Decimal(q).scaleb(-2)) for p, q in sorted(self._no.items(), reverse=True))

    @property
    def best_yes_bid(self) -> Decimal | None:
        return None if self._best_yes is None else _ticks_to_decimal(self._best_yes)

    @property
    def best_yes_ask(self) -> Decimal | None:
        return None if self._best_no is None else _ticks_to_decimal(_TICKS_PER_DOLLAR - self._best_no)

    @property
    def best_no_bid(self) -> Decimal | None:
        return None if self._best_no is None else _ticks_to_decimal(self._best_no)

    @property
    def best_no_ask(self) -> Decimal | None:
        return None if self._best_yes is None else _ticks_to_decimal(_TICKS_PER_DOLLAR - self._best_yes)

    @property
    def mid(self) -> Decimal | None:
        """YES mid-market price in dollars, or None if either side is empty."""
        ask = self.best_yes_ask_ticks
        if self._best_yes is None or ask is None:
            return None
        return Decimal(self._best_yes + ask).scaleb(-4) / 2

    @property
    def implied_probability(self) -> float | None:
        """Implied YES probability of the mid-market price, via `math_utils`."""
        ask = self.best_yes_ask_ticks
        bid = self._best_yes
        return mid_implied_probability(
            None if bid is None else bid / _TICKS_PER_DOLLAR,
            None if ask is None else ask / _TICKS_PER_DOLLAR,
        )


# --------------------------------------------------------------------------- client


def _env_name() -> Environment:
    env = os.environ.get("KALSHI_ENV", "demo")
    if env not in BASE_URLS:
        raise ValueError(f"KALSHI_ENV must be one of {sorted(BASE_URLS)}, got {env!r}")
    return env  # type: ignore[return-value]


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
        """Build with credentials and KALSHI_ENV from the environment."""
        return cls(RequestSigner.from_env(), environment=environment or _env_name(), **kwargs)

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
        return Orderbook.from_response(body, ticker)

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


# --------------------------------------------------------------------------- websocket stream

try:  # orjson parses ~3-5x faster than the stdlib on large snapshots
    import orjson

    _loads: Callable[[str | bytes], Any] = orjson.loads

    def _dumps(obj: Any) -> str:
        return orjson.dumps(obj).decode()

except ImportError:  # pragma: no cover - exercised only without orjson
    _loads = json.loads
    _dumps = json.dumps

WS_URLS: dict[str, str] = {
    "prod": "wss://external-api-ws.kalshi.com/trade-api/ws/v2",
    "demo": "wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2",
}


class KalshiStreamError(Exception):
    """Error message from the WebSocket server, or a rejected handshake."""

    def __init__(self, code: int | None, message: str):
        super().__init__(f"Kalshi stream error {code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class Quote:
    """Top of book for one market, emitted whenever the best YES bid or ask moves or resizes.

    Prices are integer ticks of $0.0001; use the Decimal properties off the hot path.
    """

    ticker: str
    bid_ticks: int | None
    ask_ticks: int | None
    bid_size: float
    ask_size: float
    implied_probability: float | None  # of the mid-market price, via math_utils
    seq: int
    exchange_ts_ms: int | None
    received_ns: int  # time.time_ns() when the frame was read

    @property
    def bid(self) -> Decimal | None:
        return None if self.bid_ticks is None else _ticks_to_decimal(self.bid_ticks)

    @property
    def ask(self) -> Decimal | None:
        return None if self.ask_ticks is None else _ticks_to_decimal(self.ask_ticks)

    @property
    def mid(self) -> Decimal | None:
        if self.bid_ticks is None or self.ask_ticks is None:
            return None
        return Decimal(self.bid_ticks + self.ask_ticks).scaleb(-4) / 2


@dataclass(frozen=True, slots=True)
class Trade:
    """A public trade print."""

    ticker: str
    trade_id: str
    yes_price: Decimal
    count: Decimal
    taker_side: str
    ts_ms: int | None
    received_ns: int


QuoteHandler = Callable[[Quote], None]
TradeHandler = Callable[[Trade], None]


class MarketStream:
    """Live order books and trades for a set of markets over one WebSocket.

    Subscribes to `orderbook_delta` (and `trade` unless disabled), keeps an `Orderbook`
    per ticker, and calls `on_quote` each time a top of book changes. Handlers run inline
    on the read loop for minimum latency, so keep them non-blocking; hand slow work to a
    task or queue.

    On a sequence gap the affected subscription is resnapshotted and its deltas are
    dropped until the snapshot lands. Dropped connections reconnect with backoff, sign a
    fresh handshake, and resubscribe. Server error messages and rejected handshakes
    (401/403) raise `KalshiStreamError` and stop the stream.
    """

    def __init__(
        self,
        signer: RequestSigner,
        market_tickers: Sequence[str],
        *,
        on_quote: QuoteHandler | None = None,
        on_trade: TradeHandler | None = None,
        environment: Environment = "demo",
        url: str | None = None,
        trades: bool = True,
        reconnect: bool = True,
        max_backoff: float = 5.0,
        ping_interval: float | None = 10.0,
        ping_timeout: float | None = 10.0,
    ):
        if not market_tickers:
            raise ValueError("market_tickers must not be empty")
        self.url = url or WS_URLS[environment]
        self.books: dict[str, Orderbook] = {t: Orderbook(t) for t in market_tickers}
        self.on_quote = on_quote
        self.on_trade = on_trade
        self._signer = signer
        self._path = urlsplit(self.url).path
        self._trades = trades
        self._reconnect = reconnect
        self._max_backoff = max_backoff
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._ws: ClientConnection | None = None
        self._closed = False
        self._next_id = 0
        self._book_sid: int | None = None
        self._last_seq: dict[int, int] = {}
        self._awaiting_snapshot: set[str] = set()  # tickers whose deltas are dropped until a snapshot
        self._tasks: set[asyncio.Task[None]] = set()
        self.gaps = 0  # sequence gaps detected, for monitoring

    @classmethod
    def from_env(cls, market_tickers: Sequence[str], environment: Environment | None = None, **kwargs: Any) -> MarketStream:
        """Build with credentials from the environment (see `RequestSigner.from_env`)."""
        return cls(RequestSigner.from_env(), market_tickers, environment=environment or _env_name(), **kwargs)

    async def run(self) -> None:
        """Connect and stream until `close()` is called or a fatal error is raised."""
        attempt = 0
        while not self._closed:
            try:
                async with ws_connect(
                    self.url,
                    additional_headers=self._signer.headers("GET", self._path),
                    compression=None,  # deflate costs CPU per frame; latency matters more than bytes
                    max_size=None,
                    ping_interval=self._ping_interval,
                    ping_timeout=self._ping_timeout,
                    open_timeout=10,
                ) as ws:
                    self._ws = ws
                    attempt = 0
                    self._reset_session()
                    await self._subscribe()
                    handle = self._handle
                    async for raw in ws:
                        handle(raw)
            except InvalidStatus as exc:
                status = exc.response.status_code
                if status in (401, 403):
                    raise KalshiStreamError(status, "handshake rejected: check API key and signature") from exc
                if not self._reconnect:
                    raise
            except (ConnectionClosed, OSError, TimeoutError):
                if self._closed:
                    return
                if not self._reconnect:
                    raise
            finally:
                self._ws = None
            if self._closed or not self._reconnect:
                return
            await asyncio.sleep(random.uniform(0, min(self._max_backoff, 0.1 * 2**attempt)))
            attempt += 1

    async def close(self) -> None:
        """Stop the stream and close the socket."""
        self._closed = True
        if self._ws is not None:
            await self._ws.close()

    # ------------------------------------------------------------------ protocol

    def _reset_session(self) -> None:
        self._book_sid = None
        self._last_seq.clear()
        self._awaiting_snapshot = set(self.books)

    def _command(self, cmd: str, params: Mapping[str, Any]) -> str:
        self._next_id += 1
        return _dumps({"id": self._next_id, "cmd": cmd, "params": params})

    async def _subscribe(self) -> None:
        assert self._ws is not None
        tickers = list(self.books)
        await self._ws.send(self._command("subscribe", {"channels": ["orderbook_delta"], "market_tickers": tickers}))
        if self._trades:
            await self._ws.send(self._command("subscribe", {"channels": ["trade"], "market_tickers": tickers}))

    def _request_snapshot(self, sid: int) -> None:
        if self._awaiting_snapshot or self._ws is None:
            return
        self._awaiting_snapshot = set(self.books)
        frame = self._command(
            "update_subscription", {"sid": sid, "action": "get_snapshot", "market_tickers": list(self.books)}
        )
        task = asyncio.get_running_loop().create_task(self._ws.send(frame))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _handle(self, raw: str | bytes) -> None:
        """Dispatch one frame. Synchronous so the read loop never yields mid-update."""
        received_ns = time.time_ns()
        frame = _loads(raw)
        kind = frame.get("type")
        if kind == "orderbook_delta":
            self._on_delta(frame, received_ns)
        elif kind == "orderbook_snapshot":
            self._on_snapshot(frame, received_ns)
        elif kind == "trade":
            self._on_trade(frame, received_ns)
        elif kind == "subscribed":
            msg = frame["msg"]
            if msg.get("channel") == "orderbook_delta":
                self._book_sid = msg["sid"]
        elif kind == "error":
            msg = frame.get("msg", {})
            raise KalshiStreamError(msg.get("code"), msg.get("msg", "unknown error"))

    def _on_snapshot(self, frame: dict[str, Any], received_ns: int) -> None:
        seq = frame["seq"]
        self._last_seq[frame["sid"]] = seq
        msg = frame["msg"]
        book = self.books.get(msg["market_ticker"])
        if book is None:
            return
        book.load(msg.get("yes_dollars_fp"), msg.get("no_dollars_fp"))
        self._awaiting_snapshot.discard(book.ticker)
        self._emit(book, seq, frame.get("sending_ts_ms"), received_ns)

    def _on_delta(self, frame: dict[str, Any], received_ns: int) -> None:
        sid, seq = frame["sid"], frame["seq"]
        last = self._last_seq.get(sid)
        self._last_seq[sid] = seq
        if last is not None and seq != last + 1 and not self._awaiting_snapshot:
            self.gaps += 1
            self._request_snapshot(sid)
            return
        msg = frame["msg"]
        book = self.books.get(msg["market_ticker"])
        if book is None or book.ticker in self._awaiting_snapshot:
            return
        if book.apply_delta(msg["side"], _parse_fixed(msg["price_dollars"], 4), _parse_fixed(msg["delta_fp"], 2)):
            self._emit(book, seq, frame.get("sending_ts_ms"), received_ns)

    def _emit(self, book: Orderbook, seq: int, exchange_ts_ms: int | None, received_ns: int) -> None:
        if self.on_quote is None:
            return
        bid, ask = book.best_yes_bid_ticks, book.best_yes_ask_ticks
        no_bid = book._best_no
        self.on_quote(
            Quote(
                ticker=book.ticker,
                bid_ticks=bid,
                ask_ticks=ask,
                bid_size=book._yes[bid] / _COUNT_SCALE if bid is not None else 0.0,
                ask_size=book._no[no_bid] / _COUNT_SCALE if no_bid is not None else 0.0,
                implied_probability=book.implied_probability,
                seq=seq,
                exchange_ts_ms=exchange_ts_ms,
                received_ns=received_ns,
            )
        )

    def _on_trade(self, frame: dict[str, Any], received_ns: int) -> None:
        if self.on_trade is None:
            return
        msg = frame["msg"]
        self.on_trade(
            Trade(
                ticker=msg["market_ticker"],
                trade_id=msg.get("trade_id", ""),
                yes_price=Decimal(msg["yes_price_dollars"]),
                count=Decimal(msg["count_fp"]),
                taker_side=msg.get("taker_side", ""),
                ts_ms=msg.get("ts_ms"),
                received_ns=received_ns,
            )
        )
