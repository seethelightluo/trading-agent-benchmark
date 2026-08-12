from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

assets = get_account_dict()["watch_list"]

def closes(sym):
    df = get_stock_daily_data(sym, days=15)
    if df is None or len(df) < 2:
        df = get_index_daily_data(sym, days=15)
    if df is None or len(df) < 2:
        return None
    return df[["date", "close"]].reset_index(drop=True)

print(f"{'sym':10s} {'prev':>12s} {'last':>12s} {'blk%':>8s}")
for a in assets:
    df = closes(a)
    if df is None:
        print(f"{a:10s} NO DATA")
        continue
    p0 = float(df["close"].iloc[0]); p1 = float(df["close"].iloc[-1])
    print(f"{a:10s} {p0:12.4f} {p1:12.4f} {(p1/p0-1)*100:7.2f}%")
