"""Print a REST order book snapshot, then stream live mid-market implied probabilities.

    python -m examples.stream_mid TICKER [TICKER ...]
"""

from __future__ import annotations

import asyncio
import sys

from kalshi_client import KalshiClient, MarketStream, Quote, Trade


def on_quote(q: Quote) -> None:
    lag = f"{q.received_ns / 1e6 - q.exchange_ts_ms:6.1f}ms" if q.exchange_ts_ms else "   n/a"
    prob = f"{q.implied_probability:.4f}" if q.implied_probability is not None else "  --  "
    print(f"{q.ticker}  bid {q.bid} x {q.bid_size:g}  ask {q.ask} x {q.ask_size:g}  p_mid {prob}  lag {lag}")


def on_trade(t: Trade) -> None:
    print(f"{t.ticker}  TRADE {t.count} @ {t.yes_price} ({t.taker_side} taker)")


async def main(tickers: list[str]) -> None:
    async with KalshiClient.from_env() as client:
        book = await client.get_orderbook(tickers[0], depth=5)
        print(f"{tickers[0]} snapshot: bid {book.best_yes_bid} ask {book.best_yes_ask} p_mid {book.implied_probability}")
    await MarketStream.from_env(tickers, on_quote=on_quote, on_trade=on_trade).run()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    try:
        asyncio.run(main(sys.argv[1:]))
    except KeyboardInterrupt:
        pass
