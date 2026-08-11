from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data
import pandas as pd

assets = list(get_account_dict()["watch_list"])

def closes(a):
    df = None
    try:
        df = get_stock_daily_data(a, days=60)
    except Exception:
        df = None
    if df is None or "close" not in df or len(df) < 30:
        try:
            df = get_index_daily_data(a, days=60)
        except Exception:
            df = None
    if df is None or "close" not in df:
        return None
    s = df[["date", "close"]].copy()
    s["date"] = pd.to_datetime(s["date"])
    return s.set_index("date")["close"].astype(float)

panel = {}
for a in assets:
    c = closes(a)
    if c is not None:
        panel[a] = c
p = pd.DataFrame(panel).sort_index()
print("last date:", p.index[-1].date(), "first:", p.index[0].date())

# block window: last 10 trading days (2027-04-08 -> 2027-04-22)
blk = p.tail(11)
rets = {}
for a in p.columns:
    s = blk[a].dropna()
    if len(s) >= 2:
        rets[a] = (s.iloc[-1] / s.iloc[0] - 1) * 100
print("\nBlock returns (10d):")
for a, r in sorted(rets.items(), key=lambda kv: kv[1]):
    print(f"  {a:>8}: {r:+.2f}%")

# 20d and 60d for context
for lbl, n in [("20d", 21), ("60d", 61)]:
    sub = p.tail(n)
    print(f"\n{lbl} returns:")
    rr = {}
    for a in p.columns:
        s = sub[a].dropna()
        if len(s) >= 2:
            rr[a] = (s.iloc[-1] / s.iloc[0] - 1) * 100
    for a, r in sorted(rr.items(), key=lambda kv: kv[1]):
        print(f"  {a:>8}: {r:+.2f}%")

# market trend check
m = p.pct_change().mean(axis=1)
print("\nmarket trend20:", round(float(m.tail(20).mean()) * 100, 3), "%")
print("market trend10:", round(float(m.tail(10).mean()) * 100, 3), "%")
print("avg level now:", round(float(p.mean(axis=1).iloc[-1]), 2))
print("avg MA60:", round(float(p.mean(axis=1).tail(60).mean()), 2))
