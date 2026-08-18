import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acct = get_account_dict()
print("total_assets:", acct.get("total_assets"))
print("net_assets:", acct.get("net_assets"))
print("available_cash:", acct.get("available_cash"))
print("market_value:", acct.get("market_value"))
print("positions:")
for p in acct.get("positions", []):
    print(f"  {p['symbol']}: qty={p['quantity']:.4f} cost={p['cost_price']:.4f} px={p['current_price']:.4f} mv={p['market_value']:.2f} pl={p['profit_loss']:.2f} plr={p['profit_loss_rate']*100:.2f}%")
print("orders:", acct.get("orders"))
print("watch_list:", acct.get("watch_list"))

print("\n--- block px change (2029-09-20 -> latest) ---")
for a in acct.get("watch_list", []):
    df = get_stock_daily_data(a, days=200)
    if df is None or len(df) < 2:
        df = get_index_daily_data(a, days=200)
    if df is None or len(df) < 2:
        print(f"  {a}: no data")
        continue
    df = df.sort_values("date")
    df["date"] = pd.to_datetime(df["date"])
    sub = df[df["date"] <= "2029-09-20"]
    if len(sub) == 0:
        print(f"  {a}: no pre-block data")
        continue
    p0 = float(sub.iloc[-1]["close"])
    p1 = float(df.iloc[-1]["close"])
    chg = (p1 / p0 - 1.0) * 100
    print(f"  {a}: {chg:+.2f}%  (p0={p0:.4f} p1={p1:.4f})")
