"""Trader probe: compare current holding weights vs strategy target as of last bar."""
import sys
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

assets = get_account_dict()["watch_list"]
N_DAYS = 300


def stock(a):
    try:
        return get_stock_daily_data(a, days=N_DAYS)
    except Exception:
        return None


def index(a):
    try:
        return get_index_daily_data(a, days=N_DAYS)
    except Exception:
        return None


frames = {a: stock(a) for a in assets}
closes = {a: (f.close.astype(float) if f is not None and "close" in f else None) for a, f in frames.items()}
print("last idx:", {a: str(c.index[-1]) for a, c in closes.items() if c is not None})

acc = get_account_dict()
mv = {p["symbol"]: p["market_value"] for p in acc["positions"]}
tot = sum(mv.values())
print("\nCURRENT HOLDING WEIGHTS:")
for s in sorted(mv, key=lambda s: -mv[s]):
    print(f"  {s:>10} {mv[s]/tot:7.4f}")
