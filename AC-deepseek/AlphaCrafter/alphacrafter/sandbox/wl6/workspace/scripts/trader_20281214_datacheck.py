"""Trader pre-cycle data check: verify data feed state, ensemble factor
computability, and current account snapshot (read-only, no orders)."""
import json
from pathlib import Path
import pandas as pd
from alphacrafter.sim.utils import (get_account_dict, get_stock_daily_data,
                                    get_index_daily_data)

acc = get_account_dict()
print("=== ACCOUNT ===")
print("date keys:", {k: acc.get(k) for k in ("total_assets", "net_assets",
      "available_cash", "market_value", "gross_position_rate")})
print("watch_list:", acc.get("watch_list"))
print("positions:")
for p in acc.get("positions", []):
    print(f"  {p['symbol']:>10} qty={p['quantity']:>12.4f} mv={p['market_value']:>14.2f} "
          f"cost={p['cost_price']:.4f} px={p['current_price']:.4f}")
print("orders:", acc.get("orders"))

assets = acc.get("watch_list", [])
print("\n=== DATA FEED ===")
for a in assets:
    df = get_stock_daily_data(a, days=300)
    if df is None or len(df) == 0:
        print(f"{a:>10}: NO DATA")
        continue
    c = df["close"].astype(float)
    ret = c.pct_change().dropna()
    last = c.iloc[-1]
    ret5 = c.iloc[-1] / c.iloc[-6] - 1 if len(c) > 6 else float("nan")
    ret20 = c.iloc[-1] / c.iloc[-21] - 1 if len(c) > 21 else float("nan")
    print(f"{a:>10}: n={len(df):>3} last={last:>12.4f} ret5={ret5:>8.2%} "
          f"ret20={ret20:>8.2%} vol20={ret.tail(20).std():>8.4f} "
          f"lastdate={df['date'].iloc[-1]}")

# Check ensemble file
for p in (Path("factor_ensemble.json"), Path("factors/factor_ensemble.json")):
    try:
        ens = json.loads(p.read_text())
        print(f"\n=== ENSEMBLE ({p}) ===")
        for f in ens.get("selected_factors", []):
            print(f"  {f['factor_id']:>22} w={f['weight']:.2f} dir={f['direction']:+d}")
    except Exception as e:
        print(p, "ERR", e)
