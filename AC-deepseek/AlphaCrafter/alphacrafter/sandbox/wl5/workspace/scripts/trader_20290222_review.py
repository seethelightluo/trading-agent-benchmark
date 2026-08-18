from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data
import pandas as pd

acc = get_account_dict()
print("=== ACCOUNT ===")
print("total_assets:", round(acc.get("total_assets", 0), 2))
print("net_assets:", round(acc.get("net_assets", 0), 2))
print("available_cash:", round(acc.get("available_cash", 0), 2))
print("market_value:", round(acc.get("market_value", 0), 2))
print("gross_position_rate:", acc.get("gross_position_rate"))
print("net_position_rate:", acc.get("net_position_rate"))
print("orders:", acc.get("orders"))
print("positions:")
for p in acc.get("positions", []):
    print(f"  {p['symbol']}: qty={p['quantity']:.4f} cost={p['cost_price']:.4f} px={p['current_price']:.4f} mv={p['market_value']:.2f} pl={p['profit_loss']:.2f} plr={p['profit_loss_rate']*100:.2f}%")

wl = acc.get("watch_list", [])
print("\n=== BLOCK RETURNS 2029-02-08 -> 2029-02-22 ===")
rows = []
for a in wl:
    df = None
    try:
        df = get_stock_daily_data(a, days=30)
    except Exception:
        df = None
    if df is None or len(df) == 0:
        try:
            df = get_index_daily_data(a, days=30)
        except Exception:
            df = None
    if df is None or len(df) == 0:
        print(f"  {a}: NO DATA")
        continue
    df = df.sort_values("date")
    df["date"] = pd.to_datetime(df["date"])
    d0 = pd.Timestamp("2029-02-08")
    sub = df[df["date"] >= d0]
    if len(sub) < 2:
        print(f"  {a}: insufficient data after block start")
        continue
    ret = sub["close"].iloc[-1] / sub["close"].iloc[0] - 1.0
    rows.append((a, ret * 100, sub["close"].iloc[0], sub["close"].iloc[-1]))
rows.sort(key=lambda x: x[1])
for a, r, p0, p1 in rows:
    print(f"  {a}: {r:+.2f}%  ({p0:.4f} -> {p1:.4f})")
print("\nmean block ret:", round(sum(r for _, r, _, _ in rows) / len(rows), 2), "%")
