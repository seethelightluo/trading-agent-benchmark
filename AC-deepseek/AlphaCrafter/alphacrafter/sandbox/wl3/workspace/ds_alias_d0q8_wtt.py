
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data
acc = get_account_dict()
OBS = {"DXY","VIX","USDCNY","USDJPY","EURUSD"}
# get close on 07-24 (decision data cutoff) and 08-08 (last day) for comparison
import pandas as pd
for p in acc.get("positions", []):
    sym = p["symbol"]
    try:
        df = get_stock_daily_data(sym, days=15) if sym not in OBS else get_index_daily_data(sym, days=15)
    except Exception:
        df = None
    if df is None or len(df) == 0:
        continue
    df = df.sort_values("date")
    last = df.iloc[-1]["close"]
    # find the row closest to 2030-07-24
    d = pd.to_datetime(df["date"])
    m = (d - pd.Timestamp("2030-07-24")).abs().argmin()
    c0724 = df.iloc[m]["close"]
    print(f"{sym:10s} cost={p['cost_price']:.4f} c0724={c0724:.4f} last={last:.4f} ratio_cost_to_0724={p['cost_price']/c0724:.4f}")
