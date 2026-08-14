"""Trader diagnostic for 2035-12-10 cycle: recent returns, VIX regime, current target."""
import json
import sys
sys.path.insert(0, '.')
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data
import strategy as S

acct = get_account_dict()
assets = list(acct["watch_list"])
print("ACCOUNT:", {k: acct[k] for k in ("total_assets", "net_assets", "available_cash",
                                        "market_value", "gross_position_rate", "net_position_rate")})
print("POSITIONS:", [(p["symbol"], round(p["quantity"], 4), p["direction"]) for p in acct.get("positions", [])])
print("OPEN ORDERS:", acct.get("orders", []))

print("\n--- Recent returns per asset (10/20/60d) ---")
for a in assets:
    df = get_index_daily_data(a, days=90)
    if df is None or len(df) < 62:
        df = None
    if df is None:
        print(f"{a:10s} NO DATA")
        continue
    c = df["close"].astype(float)
    r10 = c.iloc[-1] / c.iloc[-11] - 1 if len(c) > 10 else float("nan")
    r20 = c.iloc[-1] / c.iloc[-21] - 1 if len(c) > 20 else float("nan")
    r60 = c.iloc[-1] / c.iloc[-61] - 1 if len(c) > 60 else float("nan")
    # volatility of daily returns over last 20d annualized-ish
    vol20 = df["pct_change"].astype(float).tail(20).std() * (252 ** 0.5)
    print(f"{a:10s} r10={r10*100:7.2f}%  r20={r20*100:7.2f}%  r60={r60*100:7.2f}%  vol20={vol20*100:5.1f}%  last_close={c.iloc[-1]:.4f}")

print("\n--- VIX / macro observation ---")
for m in ("VIX", "DXY", "USDCNY", "USDJPY", "EURUSD"):
    df = get_index_daily_data(m, days=30)
    if df is not None and len(df) > 1:
        c = df["close"].astype(float)
        print(f"{m:8s} last={c.iloc[-1]:.3f}  5d={c.iloc[-1]/c.iloc[-6]-1:+.2%}  10d={c.iloc[-1]/c.iloc[-11]-1:+.2%}" if len(c) > 10 else f"{m:8s} last={c.iloc[-1]:.3f}")

print("\n--- compute_target output ---")
w, f, ids, info = S.compute_target(assets)
print("factor_ids:", ids)
print("stale:", info.get("stale"))
print("scale:", info.get("scale"))
print("weights sum:", round(sum(w.values()), 6))
for a in assets:
    print(f"  {a:10s} w={w[a]:.4f}  f={f[a]:+.5f}  score={info.get('scores', {}).get(a)}")
