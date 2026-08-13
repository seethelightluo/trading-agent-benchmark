"""Trader cycle-78 pre-flight inspection (2030-07-11 decision, visible through 07-10 close).

Checks: date state, account holdings, 20d/60d returns for all 15 tradable assets,
regime snapshot. Read-only; no orders, no date mutation.
"""
import json
from pathlib import Path

from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data

DATE_PATH = Path("../persistent/date.json")
date_state = json.loads(DATE_PATH.read_text())
print("current_date:", date_state.get("current_date"))
td = date_state.get("trading_days", [])
print("n trading days:", len(td), "| last 3:", td[-3:])

acc = get_account_dict()
print("net_assets:", acc.get("net_assets"), "| cash:", acc.get("available_cash"),
      "| gross_pos_rate:", acc.get("gross_position_rate"))
pos = {p["symbol"]: p for p in acc.get("positions", [])}
for s, p in sorted(pos.items()):
    print(f"  pos {s}: qty={p.get('quantity'):.2f} mv={p.get('market_value',0):.0f} "
          f"pl%={p.get('profit_loss_rate',0)*100:.2f}")
print("watch_list:", acc.get("watch_list"))

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

print("\n-- returns as of last visible close --")
for a in WATCH:
    try:
        df = get_stock_daily_data(a, days=130)
    except Exception:
        df = None
    if df is None or len(df) < 65:
        print(f"  {a}: NO DATA")
        continue
    c = df["close"].astype(float)
    r20 = c.iloc[-1] / c.iloc[-21] - 1.0 if len(c) >= 21 else float("nan")
    r60 = c.iloc[-1] / c.iloc[-61] - 1.0 if len(c) >= 61 else float("nan")
    r5 = c.iloc[-1] / c.iloc[-6] - 1.0 if len(c) >= 6 else float("nan")
    s20 = c.pct_change().dropna().tail(20).std()
    print(f"  {a:10s} r5={r5*100:7.2f}% r20={r20*100:7.2f}% r60={r60*100:7.2f}% vol20={s20*100:5.2f}%")

for obs in ["VIX", "DXY", "USDJPY"]:
    try:
        df = get_index_daily_data(obs, days=40)
        print(f"obs {obs}: last close {float(df['close'].iloc[-1]):.2f} (5d ago {float(df['close'].iloc[-6]):.2f})")
    except Exception as e:
        print(f"obs {obs}: err {e}")
