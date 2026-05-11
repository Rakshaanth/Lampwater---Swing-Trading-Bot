import yaml
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | None = None) -> dict:
    cfg_path = Path(path) if path else _ROOT / "config" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    for key in ("raw_dir", "features_dir", "models_dir", "logs_dir", "tickers_file"):
        if key in cfg.get("data", {}):
            cfg["data"][key] = str(_ROOT / cfg["data"][key])
    return cfg


def load_tickers(cfg: dict) -> list[str]:
    path = Path(cfg["data"]["tickers_file"])
    if not path.exists():
        raise FileNotFoundError(f"Tickers file not found: {path}. Run scripts/fetch_tickers.py first.")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]
