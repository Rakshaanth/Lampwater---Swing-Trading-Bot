"""Alpaca paper order submission + position management."""
import json
import os
from datetime import date
from pathlib import Path

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest


def trading_client() -> TradingClient:
    return TradingClient(
        os.environ.get("ALPACA_API_KEY", ""),
        os.environ.get("ALPACA_SECRET_KEY", ""),
        paper=True,
    )


def open_position_symbols(client: TradingClient) -> set[str]:
    return {p.symbol for p in client.get_all_positions()}


def select_orders(
    signals: list[tuple[str, float]],
    held: set[str],
    max_open_positions: int,
    position_size_usd: float,
) -> list[dict]:
    """Decide which buys to place. Pure function — no API calls.

    Skips symbols already held, caps total open positions, and sizes each
    order as whole shares worth up to position_size_usd. Symbols whose price
    exceeds position_size_usd (qty would be 0) are skipped.
    """
    orders = []
    slots = max_open_positions - len(held)
    for symbol, price in signals:
        if slots <= 0:
            break
        if symbol in held:
            continue
        qty = int(position_size_usd // price)
        if qty < 1:
            continue
        orders.append({"symbol": symbol, "qty": qty, "ref_price": price})
        slots -= 1
    return orders


def submit_orders(client: TradingClient, orders: list[dict], logs_dir: str) -> list[dict]:
    """Submit market buy orders (day TIF) and append results to the trade log.

    Returns the per-order results, each with status "submitted" or "error".
    """
    results = []
    for order in orders:
        record = {"ts": date.today().isoformat(), **order, "side": "buy"}
        try:
            req = MarketOrderRequest(
                symbol=order["symbol"],
                qty=order["qty"],
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            resp = client.submit_order(req)
            record.update(status="submitted", order_id=str(resp.id))
        except Exception as e:
            record.update(status="error", error=str(e))
        results.append(record)

    log_path = Path(logs_dir) / f"trades_{date.today().isoformat()}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")
    return results
