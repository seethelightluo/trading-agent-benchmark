import math
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data

acc = get_account_dict()
print("ACCOUNT net_assets:", acc.get("net_assets"), "total:", acc.get("total_assets"),
      "cash:", acc.get("available_cash"), "gross_pos:", acc.get("gross_position_rate"))
print("watch_list:", acc.get("watch_list"))
print("positions:", [(p["symbol"], round(p.get("quantity", 0), 4), round(p.get("market_value", 0), 0)) for p in acc.get("positions", [])])
print("pending orders:", len(acc.get("orders", [])))


def closes(sym, days=130, idx=True):
    fn = get_index_daily_data if idx else get_stock_daily_data
    try:
        df = fn(sym, days=days)
    except Exception as e:
        return None
    if df is None or len(df) == 0:
        return None
    df = df.sort_values("date")
    return pd.Series(df["close"].astype(float), index=pd.to_datetime(df["date"]))


assets = acc.get("watch_list", [])
print("\n=== asset state (data thru last completed day) ===")
for a in assets:
    s = closes(a, idx=False)
    if s is None or len(s) < 30:
        print(f"{a}: NO DATA via stock")
        continue
    r = s.pct_change()
    m5 = (s.iloc[-1] / s.iloc[-6] - 1) * 100
    m20 = (s.iloc[-1] / s.iloc[-21] - 1) * 100
    m60 = (s.iloc[-1] / s.iloc[-61] - 1) * 100 if len(s) > 61 else float("nan")
    ma20 = s.iloc[-1] / s.rolling(20).mean().iloc[-1]
    vol20 = r.tail(20).std() * math.sqrt(252) * 100
    print(f"{a:8s} last={s.iloc[-1]:10.2f} 5d={m5:+6.2f}% 20d={m20:+6.2f}% 60d={m60:+6.2f}% close/MA20={ma20:.3f} vol20={vol20:.1f}%")

print("\n=== macro (observation-only) ===")
for a in ["VIX", "DXY", "USDCNY", "USDJPY", "EURUSD"]:
    s = closes(a, idx=True)
    if s is None:
        print(a, "NO DATA")
        continue
    m5 = (s.iloc[-1] / s.iloc[-6] - 1) * 100
    m20 = (s.iloc[-1] / s.iloc[-21] - 1) * 100
    print(f"{a:8s} last={s.iloc[-1]:10.3f} 5d={m5:+6.2f}% 20d={m20:+6.2f}%")
