"""Check cost-price anchors: did the 08-16 proposal execute? Compare cost vs closes around 07-05 and 08-14."""
from alphacrafter.sim.utils import get_stock_daily_data

for a in ["SPX", "SX5E", "000688.SH", "US10Y"]:
    df = get_stock_daily_data(a, days=90)
    if df is None:
        print(a, "NO DATA")
        continue
    df = df.sort_values("date")
    print(f"\n{a}:")
    for _, r in df.iterrows():
        d = str(r["date"])[:10]
        if d in ("2027-07-02", "2027-07-05", "2027-07-06", "2027-08-13", "2027-08-16", "2027-08-27"):
            print(f"  {d} close={r['close']:.4f}")
