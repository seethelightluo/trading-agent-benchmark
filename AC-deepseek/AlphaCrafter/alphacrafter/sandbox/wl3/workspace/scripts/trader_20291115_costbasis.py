from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acct = get_account_dict()
OBS = {"DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"}

def get_df(sym):
    try:
        if sym in OBS:
            return get_index_daily_data(sym, days=15)
        return get_stock_daily_data(sym, days=15)
    except Exception:
        return None

pos = {p["symbol"]: p for p in acct.get("positions", [])}
print(f"{'sym':10s} {'cost':>10s} {'last3_closes':>30s}")
for sym in acct.get("watch_list", []):
    df = get_df(sym)
    p = pos.get(sym)
    if df is None or len(df) < 3:
        print(f"{sym:10s} no data")
        continue
    closes = [f"{float(x):.2f}" for x in df.sort_values('date')['close'].tail(3)]
    cost = p.get("cost_price", float('nan')) if p else float('nan')
    print(f"{sym:10s} {cost:10.4f} {' | '.join(closes):>30s}")
