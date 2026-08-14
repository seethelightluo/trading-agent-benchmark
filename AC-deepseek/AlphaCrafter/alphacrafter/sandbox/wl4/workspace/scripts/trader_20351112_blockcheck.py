"""Trader block-check: measure 10-29..11-09 block returns per asset and regime signals.
Visible data through 2035-11-09 (current date 2035-11-12)."""
import json
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

def get(a, n=170):
    try:
        f = get_stock_daily_data(a, days=n)
        if f is None:
            f = get_index_daily_data(a, days=n)
        return f
    except Exception:
        return None

print("=== Block returns (cost 10-26 close -> last visible close) ===")
block = {}
for a in ASSETS:
    f = get(a)
    if f is None or len(f) < 30:
        print(f"{a:10s} NO DATA")
        continue
    f = f.sort_values("date").reset_index(drop=True)
    last_date = f.iloc[-1]["date"]
    # find close on/before 2035-10-26
    sub = f[f["date"] <= "2035-10-26"]
    if len(sub) == 0:
        print(f"{a:10s} no pre-block row")
        continue
    ref = float(sub.iloc[-1]["close"])
    cur = float(f.iloc[-1]["close"])
    ret = cur / ref - 1.0
    block[a] = ret
    print(f"{a:10s} ref={ref:12.4f} last={cur:12.4f} ret={ret*100:+.2f}%  last_date={last_date.date()}")

print("\n=== Recent 20d momentum (last close vs 20d ago) ===")
for a in ASSETS:
    f = get(a)
    if f is None or len(f) < 25:
        continue
    f = f.sort_values("date").reset_index(drop=True)
    cur = float(f.iloc[-1]["close"])
    ref = float(f.iloc[-20]["close"])
    print(f"{a:10s} 20d ret={ (cur/ref-1)*100:+.2f}%  close={cur:.2f}")

print("\n=== Regime signals ===")
for sig in ["VIX", "DXY", "USDCNY", "USDJPY", "EURUSD"]:
    f = get_index_daily_data(sig, days=90)
    if f is None:
        f = get_stock_daily_data(sig, days=90)
    if f is None or len(f) < 5:
        print(f"{sig:8s} no data")
        continue
    f = f.sort_values("date").reset_index(drop=True)
    cur = float(f.iloc[-1]["close"])
    ref10 = float(f.iloc[-11]["close"]) if len(f) > 11 else float(f.iloc[0]["close"])
    print(f"{sig:8s} last={cur:10.3f}  10d={ (cur/ref10-1)*100:+.2f}%  last_date={f.iloc[-1]['date'].date()}")
