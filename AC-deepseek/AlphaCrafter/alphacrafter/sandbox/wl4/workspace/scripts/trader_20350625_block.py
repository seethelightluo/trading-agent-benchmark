from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data
import pandas as pd

acct = get_account_dict()


def loader(a):
    try:
        return get_stock_daily_data(a, days=120)
    except Exception:
        try:
            return get_index_daily_data(a, days=120)
        except Exception:
            return None


print("=== Block return from account (cost=block-start close, current=last close) ===")
for p in acct.get("positions", []):
    c = p.get("cost_price", 0)
    cur = p.get("current_price", 0)
    r = (cur / c - 1.0) * 100 if c else 0.0
    print(f"  {p['symbol']:10s} block_ret={r:>8.3f}%  cost={c:.4f} cur={cur:.4f}")

print("\n=== Last 20 rows per key asset (date, close) ===")
for a in ["SPX", "NDX", "SOX", "WTI", "US10Y", "N225", "SX5E", "000688.SH", "COPPER", "XAU"]:
    df = loader(a)
    if df is None:
        continue
    df = df.sort_values("date").reset_index(drop=True)
    tail = df.tail(20)
    print(f"  --- {a} ---")
    for _, r in tail.iterrows():
        print(f"    {str(r['date'].date())}  {r['close']:.4f}")

print("\n=== block return 06-10 close -> last close (per asset) ===")
for a in acct.get("watch_list", []):
    df = loader(a)
    if df is None or len(df) < 15:
        continue
    df = df.sort_values("date").reset_index(drop=True)
    dates = df["date"].astype(str)
    # find 06-10 in data
    m = dates.str.startswith("2035-06-10")
    if m.any():
        i = m.idxmax()
        r = (df.iloc[-1]["close"] / df.iloc[i]["close"] - 1.0) * 100
        print(f"  {a:10s} 06-10->last ret={r:>8.3f}%  (last={df.iloc[-1]['date'].date()})")
    else:
        # find nearest date before 06-11
        d = pd.to_datetime(dates)
        before = d[d < pd.Timestamp("2035-06-11")]
        if len(before):
            i = before.idxmax()
            r = (df.iloc[-1]["close"] / df.iloc[i]["close"] - 1.0) * 100
            print(f"  {a:10s} {df.iloc[i]['date'].date()}->last ret={r:>8.3f}%")
