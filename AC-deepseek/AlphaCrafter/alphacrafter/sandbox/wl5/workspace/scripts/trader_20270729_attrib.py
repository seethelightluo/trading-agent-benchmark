"""Attribution: 2027-07-29 -> 2027-08-12 block per-asset returns."""
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

assets = get_account_dict()["watch_list"]

def closes(a):
    df = None
    try:
        df = get_stock_daily_data(a, days=15)
    except Exception:
        df = None
    if df is None or len(df) < 11:
        try:
            df = get_index_daily_data(a, days=15)
        except Exception:
            df = None
    if df is None or len(df) < 11:
        return None
    df = df.sort_values("date")
    return df

print(f"{'asset':<10}{'px_prev':>12}{'px_now':>12}{'ret%':>9}")
for a in assets:
    df = closes(a)
    if df is None:
        print(f"{a:<10} no data")
        continue
    p0 = float(df.iloc[-11]["close"])
    p1 = float(df.iloc[-1]["close"])
    ret = (p1 / p0 - 1) * 100
    print(f"{a:<10}{p0:>12.4f}{p1:>12.4f}{ret:>9.2f}")
