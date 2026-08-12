import json, math
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acct = get_account_dict()
nav = acct.get("net_assets", 0.0)
print("nav=%.2f gross=%.2f%% cash=%.2f" % (
    nav, acct.get("gross_position_rate", 0) * 100, acct.get("available_cash", 0)))
print("positions:", len(acct.get("positions", [])))
print("pending orders:", len(acct.get("orders", [])))

OBS = {"DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"}

def get_df(sym):
    try:
        if sym in OBS:
            return get_index_daily_data(sym, days=30)
        return get_stock_daily_data(sym, days=30)
    except Exception:
        return None

pos = {p["symbol"]: p for p in acct.get("positions", [])}
rows = []
for sym in acct.get("watch_list", []):
    df = get_df(sym)
    p = pos.get(sym)
    if df is None or len(df) < 11:
        rows.append((sym, None, None, None))
        continue
    df = df.sort_values("date")
    c0 = float(df.iloc[-11]["close"])
    c1 = float(df.iloc[-1]["close"])
    r = c1 / c0 - 1.0
    mv = p.get("market_value", 0.0) if p else 0.0
    w = mv / nav if nav > 0 else 0
    rows.append((sym, r, w, mv))

rows.sort(key=lambda x: -(x[1] if x[1] is not None else 0))
print(f"{'sym':10s} {'block_ret%':>9s} {'weight%':>8s} {'mv':>12s}")
for sym, r, w, mv in rows:
    rs = f"{r*100:9.2f}" if r is not None else "      nan"
    ws = f"{w*100:8.2f}" if w is not None else "     nan"
    ms = f"{mv:12.0f}" if mv is not None else "          nan"
    print(f"{sym:10s} {rs} {ws} {ms}")
tot = sum(r * w for _, r, w, _ in rows if r is not None and w is not None)
print("approx weighted block pnl contribution: %.4f%%" % (tot * 100))
