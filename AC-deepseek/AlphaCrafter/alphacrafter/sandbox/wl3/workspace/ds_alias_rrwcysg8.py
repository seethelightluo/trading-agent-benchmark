
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data
acc = get_account_dict()
OBS = {"DXY","VIX","USDCNY","USDJPY","EURUSD"}
import pandas as pd
total = 0
rows = []
for p in acc.get("positions", []):
    sym = p["symbol"]
    try:
        df = get_stock_daily_data(sym, days=15) if sym not in OBS else get_index_daily_data(sym, days=15)
    except Exception:
        df = None
    if df is None or len(df) == 0:
        continue
    df = df.sort_values("date")
    d = pd.to_datetime(df["date"])
    m = (d - pd.Timestamp("2030-07-24")).abs().argmin()
    c0724 = df.iloc[m]["close"]
    v = p["quantity"] * c0724
    rows.append((sym, v, p["profit_loss_rate"]))
    total += v
rows.sort(key=lambda r: -r[1])
print(f"block-start notional (07-24 close): {total:.0f}")
for sym, v, pnl in rows:
    print(f"{sym:10s} w0={v/total*100:6.2f}%  block_pnl%={pnl*100:7.2f}")
