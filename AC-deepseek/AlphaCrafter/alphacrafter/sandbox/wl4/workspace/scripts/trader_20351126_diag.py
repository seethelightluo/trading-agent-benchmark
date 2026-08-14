"""Trader diagnostic 2035-11-26: block returns for recent windows + regime signals
+ proposed target under the new 0.55/0.45 ensemble."""
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

def window_ret(f, end, start):
    f = f.sort_values("date").reset_index(drop=True)
    sub_e = f[f["date"] <= end]
    sub_s = f[f["date"] <= start]
    if len(sub_e) == 0 or len(sub_s) == 0:
        return None
    return float(sub_e.iloc[-1]["close"]) / float(sub_s.iloc[-1]["close"]) - 1.0

print("=== last visible dates ===")
for a in ASSETS:
    f = get(a)
    if f is None:
        print(f"{a:10s} NO DATA")
        continue
    f = f.sort_values("date").reset_index(drop=True)
    print(f"{a:10s} last={f.iloc[-1]['date'].date()}")

print("\n=== block returns (10-29..11-09 and 11-09..last) ===")
for a in ASSETS:
    f = get(a)
    if f is None:
        continue
    r1 = window_ret(f, "2035-11-09", "2035-10-29")
    r2 = window_ret(f, "2035-11-25", "2035-11-09")
    print(f"{a:10s} 10-29..11-09={r1*100:+.2f}%  11-09..last={r2*100:+.2f}%")

print("\n=== regime signals ===")
for sig in ["VIX", "DXY", "USDCNY", "USDJPY", "EURUSD"]:
    f = get_index_daily_data(sig, days=120)
    if f is None:
        f = get_stock_daily_data(sig, days=120)
    if f is None or len(f) < 5:
        print(f"{sig:8s} no data")
        continue
    f = f.sort_values("date").reset_index(drop=True)
    cur = float(f.iloc[-1]["close"])
    ref21 = float(f.iloc[-22]["close"]) if len(f) > 22 else float(f.iloc[0]["close"])
    ref5 = float(f.iloc[-6]["close"]) if len(f) > 6 else ref21
    print(f"{sig:8s} last={cur:10.3f}  5d={ (cur/ref5-1)*100:+.2f}%  21d={ (cur/ref21-1)*100:+.2f}%  last_date={f.iloc[-1]['date'].date()}")

print("\n=== proposed target under current ensemble ===")
import sys
sys.path.insert(0, ".")
import strategy as st
acct = st.get_account_dict()
assets = list(acct["watch_list"])
w, f, ids, info = st.compute_target(assets)
print("factor_ids:", ids)
print("weights sum:", round(sum(w.values()), 6))
for a in assets:
    print(f"  {a:10s} w={w[a]:.4f} f={f[a]:+.5f}")
print("info:", json.dumps(info)[:800])
