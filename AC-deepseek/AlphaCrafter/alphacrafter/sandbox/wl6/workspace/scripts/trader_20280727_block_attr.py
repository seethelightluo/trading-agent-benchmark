from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

assets = list(get_account_dict()["watch_list"])
print("WATCHLIST:", assets)
print("\nBlock 2028-07-13 -> 2028-07-27 returns (close-to-close, 10 trading days):")
for a in assets:
    f = None
    try:
        f = get_stock_daily_data(a, days=30)
    except Exception:
        f = None
    if f is None or len(f) < 12:
        try:
            f = get_index_daily_data(a, days=30)
        except Exception:
            f = None
    if f is None or len(f) < 12:
        print(f"{a:10s} no data")
        continue
    f = f.sort_values("date")
    c = f["close"].astype(float)
    # last 11 closes => 10 daily returns covering block start..end
    seg = c.tail(11)
    r = (seg.iloc[-1] / seg.iloc[0] - 1.0) * 100
    # also 20d for context
    seg20 = c.tail(21)
    r20 = (seg20.iloc[-1] / seg20.iloc[0] - 1.0) * 100 if len(seg20) == 21 else float("nan")
    print(f"{a:10s} block10d={r:7.2f}%   20d={r20:7.2f}%   last_close={seg.iloc[-1]:.2f}")
