"""Pull historical OHLCV for all tickers and save to data/raw/."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.config import load_config, load_tickers
from src.data import fetch_all

cfg = load_config()
tickers = load_tickers(cfg)

years = cfg["data"]["history_years"]
end = datetime.today()
start = end - timedelta(days=years * 365)

print(f"Fetching {len(tickers)} tickers from {start.date()} to {end.date()}")
fetch_all(tickers, start, end, cfg["data"]["raw_dir"])
print("Done.")
