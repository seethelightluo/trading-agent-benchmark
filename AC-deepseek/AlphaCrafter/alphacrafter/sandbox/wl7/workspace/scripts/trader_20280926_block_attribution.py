"""Block attribution for 2028-09-26 -> 2028-10-10 live cycle."""
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acc = get_account_dict()
pos = {p["symbol"]: p for p in acc["positions"]}
total = acc["total_assets"]

start_date = "2028-09-26"
end_date = "2028-10-10"

def price_on(sym, date):
    df = get_stock_daily_data(sym, days=300)
    if df is None or len(df) == 0:
        df = get_index_daily_data(sym, days=300)
    if df is None or len(df) == 0:
        return None, None
    df = df.sort_values("date")
    ds = [str(d)[:10] for d in df["date"]]
    try:
        i0 = ds.index(start_date)
    except ValueError:
        i0 = None
    try:
        i1 = ds.index(end_date)
    except ValueError:
        i1 = None
    p0 = df.iloc[i0]["close"] if i0 is not None else df.iloc[0]["close"]
    p1 = df.iloc[i1]["close"] if i1 is not None else df.iloc[-1]["close"]
    return float(p0), float(p1)

print(f"{'asset':10s} {'qty':>10s} {'wt%':>6s} {'ret%':>8s} {'pnl$':>10s} {'pp':>6s}")
tot_pnl = 0.0
for sym, p in pos.items():
    qty = p["quantity"]
    mv = p["market_value"]
    wt = mv / total * 100
    p0, p1 = price_on(sym, start_date)
    if p0 is None or p1 is None:
        print(f"{sym:10s} {qty:10.3f} {wt:6.2f}  n/a")
        continue
    ret = (p1 / p0 - 1) * 100
    pnl = qty * (p1 - p0)
    pp = pnl / total * 100
    tot_pnl += pnl
    print(f"{sym:10s} {qty:10.3f} {wt:6.2f} {ret:8.2f} {pnl:10.0f} {pp:6.2f}")

print(f"\nSum attribution pnl = {tot_pnl:.0f} ({tot_pnl/total*100:.2f}% of NAV {total:.0f})")
