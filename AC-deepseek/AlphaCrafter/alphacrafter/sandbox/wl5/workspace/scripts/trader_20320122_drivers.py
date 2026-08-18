import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acc = get_account_dict()
positions = {p["symbol"]: p for p in acc.get("positions", [])}
watch = acc.get("watch_list", [])

def closes(sym, days=40):
    df = None
    try:
        df = get_stock_daily_data(sym, days=days)
    except Exception:
        df = None
    if df is None or len(df) < 2:
        try:
            df = get_index_daily_data(sym, days=days)
        except Exception:
            df = None
    if df is None:
        return None
    s = df[["date", "close"]].copy()
    s["date"] = pd.to_datetime(s["date"])
    return s.set_index("date")["close"].astype(float)

start = pd.Timestamp("2032-01-08")
tot_pl = 0.0
for s in watch:
    c = closes(s)
    if c is None:
        print(s, "no data")
        continue
    seg = c[c.index >= start]
    if len(seg) < 2:
        print(s, "insufficient", len(seg))
        continue
    p0 = float(seg.iloc[0])
    p1 = float(seg.iloc[-1])
    chg = p1 / p0 - 1.0
    qty = positions.get(s, {}).get("quantity", 0.0)
    pl = qty * p0 * chg
    tot_pl += pl
    print(f"{s}: {p0:.2f} -> {p1:.2f} ({chg*100:+.2f}%) qty={qty:.4f} contrib={pl:+,.0f} (mv0={qty*p0:,.0f})")
print(f"TOTAL approx block PnL: {tot_pl:+,.0f}")
