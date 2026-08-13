"""miner_1 datacheck (2032-02-09): data availability, volume, macro signals, stale windows."""
import sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from factor_research_lib import load_panels, close_panel, TRADABLE, MACRO

panels = load_panels(days=3500)
closes = close_panel(panels)
rets = closes.pct_change()

print("=== close coverage (last 700d) per asset ===")
recent = closes.tail(700)
for a in closes.columns:
    n_valid = recent[a].notna().sum()
    last = recent[a].dropna().index[-1].date()
    flat = recent[a].dropna()
    n_flat = int((flat.diff().abs() < 1e-12).sum()) if len(flat) > 5 else 0
    print(f"{a:10s} valid={n_valid:4d}/700 last={last} zero_change_days={n_flat}")

print("\n=== volume availability (last 700d) ===")
for a in TRADABLE:
    df = panels.get(a)
    if df is not None and "volume" in df.columns:
        v = df["volume"].tail(700)
        print(f"{a:10s} volume_valid={v.notna().sum():4d}/700 last={v.dropna().index[-1].date() if v.notna().any() else None}")

print("\n=== macro signals (last 700d) ===")
for m in MACRO:
    df = panels.get(m)
    if df is not None:
        c = df["close"].tail(700)
        print(f"{m:10s} valid={c.notna().sum():4d}/700 last={c.dropna().index[-1].date() if c.notna().any() else None}")

print("\n=== recent 120d close tail (CN10Y, US10Y, BTC, ETH) ===")
for a in ["CN10Y", "US10Y", "BTC", "ETH", "000300.SH", "HSI"]:
    s = closes[a].tail(120)
    s = s[s.notna()]
    print(f"--- {a} n={len(s)} first={s.index[0].date()} last={s.index[-1].date()}")
    print(s.tail(8).round(4).to_string())
