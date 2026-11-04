import asyncio
import base64
import json
from decimal import Decimal

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519
from websockets.asyncio.server import serve

from kalshi_client import KalshiStreamError, MarketStream, Orderbook, RequestSigner, _parse_fixed
from math_utils import mid_implied_probability, mid_price

T = "FED-23DEC-T3.00"


# ---------------------------------------------------------------- math_utils


def test_mid_implied_probability():
    assert mid_price(0.42, 0.45) == pytest.approx(0.435)
    assert mid_implied_probability(0.42, 0.45) == pytest.approx(0.435)
    assert mid_implied_probability(None, 0.45) is None
    with pytest.raises(ValueError):
        mid_price(0.5, 0.4)


# ---------------------------------------------------------------- order book


def test_parse_fixed_is_exact():
    assert _parse_fixed("0.9600", 4) == 9600
    assert _parse_fixed("0.0525", 4) == 525
    assert _parse_fixed("-54.00", 2) == -5400
    assert _parse_fixed("3", 2) == 300
    assert _parse_fixed("0.560000", 4) == 5600
    with pytest.raises(ValueError):
        _parse_fixed("0.12345", 4)


def test_deltas_track_top_of_book():
    book = Orderbook(T)
    book.load([["0.4000", "5.00"], ["0.4200", "1.00"]], [["0.5500", "3.00"]])
    assert book.best_yes_bid == Decimal("0.42") and book.best_yes_ask == Decimal("0.45")
    assert book.mid == Decimal("0.435")
    assert book.implied_probability == pytest.approx(0.435)

    assert book.apply_delta("yes", 4000, 200) is False  # below the top: no quote change
    assert book.apply_delta("yes", 4200, 100) is True  # top resized
    assert book.apply_delta("yes", 4300, 100) is True  # new best bid
    assert book.best_yes_bid_ticks == 4300
    assert book.apply_delta("yes", 4300, -100) is True  # best removed, falls back
    assert book.best_yes_bid_ticks == 4200
    assert book.apply_delta("no", 5500, -300) is True  # NO side empties
    assert book.best_yes_ask is None and book.implied_probability is None
    assert book.size_at("yes", 4000) == Decimal("7.00")


# ---------------------------------------------------------------- live stream against a local server


def snapshot(sid, seq, yes, no):
    return {"type": "orderbook_snapshot", "sid": sid, "seq": seq, "sending_ts_ms": 1,
            "msg": {"market_ticker": T, "yes_dollars_fp": yes, "no_dollars_fp": no}}


def delta(sid, seq, side, price, qty):
    return {"type": "orderbook_delta", "sid": sid, "seq": seq, "sending_ts_ms": 2,
            "msg": {"market_ticker": T, "side": side, "price_dollars": price, "delta_fp": qty}}


def run_stream(server_script, **kwargs):
    """Run a MarketStream against a local server; return (quotes, trades, stream, commands)."""
    key = ed25519.Ed25519PrivateKey.generate()
    quotes, trades, commands = [], [], []

    async def handler(ws):
        headers = ws.request.headers
        message = f"{headers['KALSHI-ACCESS-TIMESTAMP']}GET/trade-api/ws/v2".encode()
        key.public_key().verify(base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"]), message)
        await server_script(ws, commands)

    async def main():
        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            stream = MarketStream(
                RequestSigner("k", key), [T], url=f"ws://127.0.0.1:{port}/trade-api/ws/v2",
                on_quote=quotes.append, on_trade=trades.append, reconnect=False, **kwargs,
            )
            await asyncio.wait_for(stream.run(), 5)
            return stream

    stream = asyncio.run(main())
    return quotes, trades, stream, commands


async def recv_json(ws, commands):
    cmd = json.loads(await ws.recv())
    commands.append(cmd)
    return cmd


def test_stream_emits_quotes_with_implied_probability():
    async def script(ws, commands):
        book_cmd = await recv_json(ws, commands)
        await recv_json(ws, commands)  # trade subscription
        await ws.send(json.dumps({"type": "subscribed", "id": book_cmd["id"], "msg": {"channel": "orderbook_delta", "sid": 2}}))
        await ws.send(json.dumps(snapshot(2, 1, [["0.4200", "10.00"]], [["0.5500", "3.00"]])))
        await ws.send(json.dumps(delta(2, 2, "yes", "0.3000", "5.00")))  # deep level: no quote
        await ws.send(json.dumps(delta(2, 3, "yes", "0.4400", "2.00")))  # new best bid
        await ws.send(json.dumps({"type": "trade", "sid": 3, "seq": 1, "msg": {
            "market_ticker": T, "trade_id": "t1", "yes_price_dollars": "0.4400", "count_fp": "2.00",
            "taker_side": "yes", "ts_ms": 5}}))

    quotes, trades, stream, commands = run_stream(script)
    assert commands[0]["params"] == {"channels": ["orderbook_delta"], "market_tickers": [T]}
    assert commands[1]["params"]["channels"] == ["trade"]
    assert len(quotes) == 2
    first, second = quotes
    assert (first.bid, first.ask) == (Decimal("0.42"), Decimal("0.45"))
    assert first.implied_probability == pytest.approx(0.435)
    assert first.bid_size == 10.0 and first.ask_size == 3.0
    assert second.bid == Decimal("0.44") and second.mid == Decimal("0.445")
    assert second.implied_probability == pytest.approx(0.445)
    assert second.received_ns >= first.received_ns
    assert trades[0].yes_price == Decimal("0.44") and trades[0].count == Decimal("2.00")
    assert stream.books[T].size_at("yes", 3000) == Decimal("5.00")


def test_sequence_gap_requests_snapshot_and_drops_stale_deltas():
    async def script(ws, commands):
        await recv_json(ws, commands)
        await recv_json(ws, commands)
        await ws.send(json.dumps(snapshot(2, 1, [["0.4200", "10.00"]], [["0.5500", "3.00"]])))
        await ws.send(json.dumps(delta(2, 3, "yes", "0.4600", "1.00")))  # seq 2 missing
        resync = await recv_json(ws, commands)
        assert resync["cmd"] == "update_subscription"
        assert resync["params"]["action"] == "get_snapshot" and resync["params"]["sid"] == 2
        await ws.send(json.dumps(delta(2, 4, "yes", "0.4700", "1.00")))  # dropped while resyncing
        await ws.send(json.dumps(snapshot(2, 5, [["0.4800", "4.00"]], [["0.5000", "1.00"]])))
        await ws.send(json.dumps(delta(2, 6, "no", "0.5100", "2.00")))  # applied again

    quotes, _, stream, _ = run_stream(script)
    assert stream.gaps == 1
    assert [q.bid for q in quotes] == [Decimal("0.42"), Decimal("0.48"), Decimal("0.48")]
    assert quotes[-1].ask == Decimal("0.49")
    assert stream.books[T].best_yes_bid == Decimal("0.48")


def test_server_error_raises():
    async def script(ws, commands):
        await recv_json(ws, commands)
        await ws.send(json.dumps({"type": "error", "id": 1, "msg": {"code": 8, "msg": "Unknown channel name"}}))
        await asyncio.sleep(1)

    with pytest.raises(KalshiStreamError) as exc:
        run_stream(script, trades=False)
    assert exc.value.code == 8


def test_reconnects_and_resubscribes():
    key = ed25519.Ed25519PrivateKey.generate()
    connections, quotes = [], []

    async def main():
        async def handler(ws):
            connections.append(ws.request.headers["KALSHI-ACCESS-SIGNATURE"])
            await ws.recv()
            await ws.recv()
            await ws.send(json.dumps(snapshot(2, 1, [["0.4200", "1.00"]], [["0.5500", "1.00"]])))
            if len(connections) == 1:
                return  # drop the first connection
            await ws.wait_closed()

        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            stream = MarketStream(RequestSigner("k", key), [T], url=f"ws://127.0.0.1:{port}/trade-api/ws/v2",
                                  on_quote=quotes.append, max_backoff=0.05)
            task = asyncio.create_task(stream.run())
            while len(quotes) < 2:
                await asyncio.sleep(0.01)
            await stream.close()
            await asyncio.wait_for(task, 2)

    asyncio.run(asyncio.wait_for(main(), 5))
    assert len(connections) == 2
    assert len(quotes) == 2
