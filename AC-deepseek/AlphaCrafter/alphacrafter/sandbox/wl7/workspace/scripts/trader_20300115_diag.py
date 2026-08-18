"""Trader diagnostic 2030-01-15: verify data availability + strategy factor branch."""
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acc = get_account_dict()
assets = acc.get("watch_list", [])
print("watch_list:", assets)
print("num assets:", len(assets))

for a in assets:
    try:
        df = get_stock_daily_data(a, days=200)
        if df is None or len(df) == 0:
            print(f"  {a}: NO DATA (stock api)")
        else:
            last = df.iloc[-1]
            print(f"  {a}: rows={len(df)} last_date={last['date']} close={last['close']:.4f}")
    except Exception as e:
        print(f"  {a}: stock api EXC {e}")

for a in ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]:
    try:
        df = get_index_daily_data(a, days=200)
        if df is None or len(df) == 0:
            print(f"  {a}: NO DATA (index api)")
        else:
            last = df.iloc[-1]
            print(f"  {a}: rows={len(df)} last_date={last['date']} close={last['close']:.4f}")
    except Exception as e:
        print(f"  {a}: index api EXC {e}")
