"""Block return attribution for 2028-12-14 -> 2028-12-28 cycle."""
import json
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict

acc = get_account_dict()
assets = list(acc["watch_list"])
pos = {p["symbol"]: p for p in acc.get("positions", [])}
total = acc.get("net_assets", 0.0)

rows = []
for a in assets:
    df = None
    try:
        df = get_stock_daily_data(a, days=15)
    except Exception:
        df = None
    if df is None or len(df) < 2:
        try:
            df = get_index_daily_data(a, days=15)
        except Exception:
            df = None
    if df is None or len(df) < 2:
        continue
    df = df.sort_values("date")
    p0 = float(df.iloc[-11]["close"])   # close before block start
    p1 = float(df.iloc[-1]["close"])    # latest close
    ret = p1 / p0 - 1.0
    mv = pos.get(a, {}).get("market_value", 0.0)
    contrib = mv * ret / total * 100.0
    rows.append((a, ret * 100.0, mv / total * 100.0, contrib, pos.get(a, {}).get("profit_loss", 0.0)))

rows.sort(key=lambda r: -r[1])
print(f"{'asset':<10}{'block_ret%':>10}{'w%':>8}{'contrib_pp':>11}{'cum_pl':>10}")
for a, r, w, c, pl in rows:
    print(f"{a:<10}{r:>10.2f}{w:>8.2f}{c:>11.2f}{pl:>10.2f}")
print(f"\nnet_assets: {total:.2f}")
print(f"sum contrib: {sum(r[3] for r in rows):.2f} pp")
