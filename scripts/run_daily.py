"""Daily rule-based run: scan all tickers for entry signals, optionally trade.

Usage:
  python scripts/run_daily.py --no-trade            # dry run, fresh data
  python scripts/run_daily.py --no-trade --cached   # dry run on data/raw/ cache
  python scripts/run_daily.py                       # live paper trading run
"""
import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.config import load_config, load_tickers
from src.data import fetch_ohlcv, load_ohlcv
from src.signals import scan_all


def get_data(tickers: list[str], cfg: dict, cached: bool) -> dict:
    data = {}
    if cached:
        for symbol in tickers:
            try:
                data[symbol] = load_ohlcv(symbol, cfg["data"]["raw_dir"])
            except FileNotFoundError:
                print(f"  {symbol}: no cached data, skipping")
        return data

    # 400 calendar days ≈ 270 trading days, enough for the 200d SMA + warmup
    end = datetime.today()
    start = end - timedelta(days=400)
    for symbol in tickers:
        try:
            df = fetch_ohlcv(symbol, start, end)
        except Exception as e:
            print(f"  {symbol}: fetch error: {e}")
            continue
        if df.empty:
            print(f"  {symbol}: no data returned")
        else:
            data[symbol] = df
        time.sleep(0.3)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-trade", action="store_true", help="scan and print signals, submit no orders")
    parser.add_argument("--cached", action="store_true", help="use data/raw/ cache instead of fetching fresh bars")
    args = parser.parse_args()

    cfg = load_config()
    tickers = load_tickers(cfg)

    print(f"Loading data for {len(tickers)} tickers ({'cache' if args.cached else 'live fetch'})...")
    data = get_data(tickers, cfg, cached=args.cached)
    print(f"Scanning {len(data)} tickers (trigger_mode={cfg['labeling']['trigger_mode']})...")

    signals = scan_all(data, cfg)
    if signals.empty:
        print("No entry signals today.")
        return
    print(f"\n{len(signals)} signal(s):\n{signals.round(2).to_string()}\n")

    if args.no_trade:
        print("Dry run (--no-trade): no orders submitted.")
        return

    from src.execute import open_position_symbols, select_orders, submit_orders, trading_client

    client = trading_client()
    held = open_position_symbols(client)
    print(f"Currently holding {len(held)} position(s): {sorted(held) or '—'}")

    orders = select_orders(
        list(zip(signals.index, signals["close"])),
        held,
        cfg["execution"]["max_open_positions"],
        cfg["execution"]["position_size_usd"],
    )
    if not orders:
        print("No orders to place (all signals held, or position cap reached).")
        return

    results = submit_orders(client, orders, cfg["data"]["logs_dir"])
    for r in results:
        if r["status"] == "submitted":
            print(f"  BUY {r['qty']} {r['symbol']} @ ~{r['ref_price']:.2f} — order {r['order_id']}")
        else:
            print(f"  BUY {r['qty']} {r['symbol']} FAILED: {r['error']}")
    print(f"Trade log: {cfg['data']['logs_dir']}/trades_{datetime.today().date()}.jsonl")


if __name__ == "__main__":
    main()
