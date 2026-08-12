from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data
import pandas as pd

acct = get_account_dict()
OBS = {"DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"}

def get_df(sym):
    try:
        if sym in OBS:
            return get_index_daily_data(sym, days=40)
        return get_stock_daily_data(sym, days=40)
    except Exception:
        return None

pos = {p["symbol"]: p for p in acct.get("positions", [])}
# find dates in the window 10-28..11-04 to compare with cost
for sym in ["SPX", "XAU", "COPPER", "000300.SH", "ETH"]:
    df = get_df(sym)
    p = pos.get(sym)
    if df is None:
        continue
    df = df.sort_values("date")
    sub = df[(df["date"] >= "2029-10-29") & (df["date"] <= "2029-11-04")]
    cost = p.get("cost_price", float('nan')) if p else float('nan')
    print(f"--- {sym} cost={cost:.4f}")
    for _, r in sub.iterrows():
        print(f"    {r['date'].date()} close={float(r['close']):.4f}")
