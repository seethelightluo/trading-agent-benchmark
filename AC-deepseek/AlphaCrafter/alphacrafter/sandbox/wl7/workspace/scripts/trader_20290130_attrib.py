from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acc = get_account_dict()
assets = acc["watch_list"]
pos = {p["symbol"]: p for p in acc["positions"]}
mv = {p["symbol"]: p["market_value"] for p in acc["positions"]}
total = sum(mv.values())

# block start price: need close ~10 trading days before last bar (last bar = 02-13)
def price_series(sym):
    df = get_stock_daily_data(sym, days=15)
    if df is None or len(df) < 2:
        df = get_index_daily_data(sym, days=15)
    if df is None or len(df) < 2:
        return None
    df = df.sort_values("date")
    return df

print("date range check (first asset):")
d0 = price_series(assets[0])
print("  first date", d0["date"].iloc[0], "last date", d0["date"].iloc[-1], "rows", len(d0))

attrib = {}
for a in assets:
    df = price_series(a)
    if df is None or len(df) < 2:
        print(a, "NO DATA")
        continue
    close = df["close"].astype(float)
    p_start = close.iloc[-11] if len(close) >= 11 else close.iloc[0]
    p_end = close.iloc[-1]
    r = p_end / p_start - 1.0
    w = mv.get(a, 0) / total
    attrib[a] = (r, w, r * w)
    print("%-10s ret=%8.2f%% wt=%6.2f%% contrib=%+7.2fpp" % (a, r * 100, w * 100, r * w * 100))

print("\nsum contrib pp:", round(sum(v[2] for v in attrib.values()) * 100, 2))
