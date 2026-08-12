from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

vix = get_index_daily_data("VIX", days=60)
if vix is not None:
    vix = vix.sort_values("date")
    print("VIX last close:", round(float(vix.iloc[-1]["close"]), 1),
          "| 21d ago:", round(float(vix.iloc[-22]["close"]), 1) if len(vix) > 22 else None)
    print("VIX last 5 closes:", [round(float(x), 1) for x in vix["close"].tail(5)])

for sym in ["SPX", "XAU", "COPPER", "000300.SH", "WTI", "N225", "SOX"]:
    df = get_stock_daily_data(sym, days=120)
    if df is None or len(df) < 62:
        print(f"{sym}: insufficient data")
        continue
    df = df.sort_values("date")
    c = df["close"].astype(float)
    r60 = float(c.iloc[-1]) / float(c.iloc[-61]) - 1
    r20 = float(c.iloc[-1]) / float(c.iloc[-21]) - 1
    print(f"{sym}: 60d={r60*100:6.1f}%  20d={r20*100:6.1f}%")
