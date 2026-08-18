"""miner_3 2026-12-03: check data availability and freshness."""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data

WATCHLIST = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
             "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]


def load_asset(symbol, days=1500):
    df = None
    try:
        df = get_index_daily_data(symbol=symbol, days=days)
    except Exception:
        df = None
    if df is None:
        try:
            df = get_stock_daily_data(symbol=symbol, days=days)
        except Exception:
            df = None
    if df is None:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


print("=== ASSET FRESHNESS (last 3 rows, last date, n) ===")
panels = {}
for s in WATCHLIST:
    df = load_asset(s, days=1500)
    if df is None or len(df) == 0:
        print(f"{s:10s} NO DATA")
        continue
    panels[s] = df
    last = df["date"].iloc[-1].date()
    print(f"{s:10s} last={last} n={len(df):5d} last_close={df['close'].iloc[-1]:.2f} "
          f"prev_close={df['close'].iloc[-2]:.2f} prev2={df['close'].iloc[-3]:.2f}")

print("\n=== MACRO FRESHNESS ===")
for m in MACRO:
    try:
        df = pd.read_csv(f"../persistent/index_data/{m}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        print(f"{m:8s} last={df['date'].iloc[-1].date()} n={len(df)} last_close={df['close'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"{m:8s} ERROR {e}")

# Build common panel dates and check frozen assets
print("\n=== COMMON DATE RANGE / STALE CHECK ===")
closes = {}
for s, df in panels.items():
    closes[s] = df.set_index("date")["close"]
panel = pd.DataFrame(closes).sort_index()
print(f"panel dates: {panel.index.min().date()} .. {panel.index.max().date()}  shape={panel.shape}")
# per-asset last valid date
for s in panel.columns:
    sub = panel[s].dropna()
    print(f"{s:10s} last valid {sub.index[-1].date()}  flat_last60={np.isclose(sub.iloc[-1], sub.iloc[-61], equal_nan=True) if len(sub) > 61 else 'n/a'}")