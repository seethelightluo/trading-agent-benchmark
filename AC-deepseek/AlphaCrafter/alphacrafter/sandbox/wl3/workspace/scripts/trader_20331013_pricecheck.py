"""Check whether rebalance executed at 09-29 close: compare position cost vs close prices."""
from alphacrafter.sim.utils import get_stock_daily_data

for sym in ["WTI", "ETH", "COPPER", "SPX", "NDX", "000688.SH", "XAU", "N225"]:
    df = get_stock_daily_data(symbol=sym, days=40)
    if df is None:
        print(sym, "no data"); continue
    df = df.sort_values("date")
    # find rows around 09-15, 09-29
    sub = df[df["date"] >= "2033-09-13"]
    for _, r in sub.iterrows():
        d = r["date"].strftime("%Y-%m-%d")
        if d in ("2033-09-15", "2033-09-16", "2033-09-28", "2033-09-29", "2033-09-30", "2033-10-02"):
            print(f"{sym:10s} {d} close={r['close']:.4f}")
    print()
