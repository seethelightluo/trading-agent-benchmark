"""Trader cycle review 2027-08-16..2027-08-30: account state + per-asset block returns."""
import json
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acc = get_account_dict()
print("net_assets:", round(acc.get("net_assets", 0), 2))
print("available_cash:", round(acc.get("available_cash", 0), 2))
print("market_value:", round(acc.get("market_value", 0), 2))
print("total_profit_loss:", round(acc.get("total_profit_loss", 0), 2))
print("total_profit_loss_rate:", round(acc.get("total_profit_loss_rate", 0), 4))
print("gross_position_rate:", round(acc.get("gross_position_rate", 0), 4))
print("positions:")
for p in acc.get("positions", []):
    print(f"  {p['symbol']:8s} qty={p['quantity']:12.4f} cost={p['cost_price']:.4f} "
          f"px={p['current_price']:.4f} mktval={p['market_value']:12.2f} "
          f"pnl={p['profit_loss']:12.2f} pnl%={p['profit_loss_rate']*100:7.3f}%")

wl = acc.get("watch_list", [])
print("\nwatch_list:", wl)

# per-asset block return: close at ~08-14 (before block) vs last close (08-28/08-29)
print("\nper-asset block return (close ~08-14 -> last close):")
for a in wl:
    try:
        df = get_stock_daily_data(a, days=14)
    except Exception:
        df = None
    if df is None or len(df) < 3:
        try:
            df = get_index_daily_data(a, days=14)
        except Exception:
            df = None
    if df is None or len(df) < 3:
        print(f"  {a:8s} NO DATA")
        continue
    df = df.sort_values("date")
    first = df.iloc[0]["close"]
    last = df.iloc[-1]["close"]
    ret = (last - first) / first * 100
    print(f"  {a:8s} {str(df.iloc[0]['date'])[:10]}->{str(df.iloc[-1]['date'])[:10]} "
          f"ret={ret:+.2f}% n={len(df)}")
