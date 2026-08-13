"""Trader 2033-04-18: analyze the 04-05..04-18 (drifted) block returns vs 03-21 target."""
import json
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data

acct = get_account_dict()
assets = acct["watch_list"]
nav = acct["total_assets"]

# find last close before block start (04-04) and latest visible (04-15)
def get_df(a):
    try:
        return get_stock_daily_data(a, days=60)
    except Exception:
        try:
            return get_index_daily_data(a, days=60)
        except Exception:
            return None

rets = {}
for a in assets:
    df = get_df(a)
    if df is None or len(df) < 30:
        rets[a] = None
        continue
    df = df.sort_values("date")
    dts = [str(d.date()) for d in df["date"]]
    if "2033-04-04" in dts and "2033-04-15" in dts:
        p0 = float(df.loc[df["date"].astype(str).str.slice(0, 10) == "2033-04-04", "close"].iloc[0])
        p1 = float(df.loc[df["date"].astype(str).str.slice(0, 10) == "2033-04-15", "close"].iloc[0])
        rets[a] = p1 / p0 - 1.0
    elif "2033-04-18" in dts:
        p0 = float(df.loc[df["date"].astype(str).str.slice(0, 10) == "2033-04-04", "close"].iloc[0]) if "2033-04-04" in dts else None
        p1 = float(df.loc[df["date"].astype(str).str.slice(0, 10) == "2033-04-18", "close"].iloc[0])
        if p0:
            rets[a] = p1 / p0 - 1.0
    else:
        rets[a] = None

# position values for block attribution (approx using current qty * block ret)
pos = {p["symbol"]: p for p in acct.get("positions", [])}
print(f"NAV {nav:.2f}  cash {acct['available_cash']:.2f}  gross_pos {acct['gross_position_rate']:.4f}")
tot_mv = sum(p["market_value"] for p in acct["positions"])
print(f"{'asset':10s} {'block_ret':>10s} {'w_now':>8s} {'mv':>12s}")
contrib = 0.0
for a in assets:
    mv = pos.get(a, {}).get("market_value", 0.0)
    r = rets.get(a)
    w = mv / nav if nav else 0
    c = w * r if r is not None else 0.0
    contrib += c
    print(f"{a:10s} {('%.4f' % r) if r is not None else 'n/a':>10s} {w:8.4f} {mv:12.2f}  contrib~{c:+.5f}")
print(f"sum contrib ~ {contrib:+.5f} (approx block PnL share)")
print("account nav: last cycle end 1118545.22 -> now", nav, "drift %", (nav/1118545.22-1)*100)
