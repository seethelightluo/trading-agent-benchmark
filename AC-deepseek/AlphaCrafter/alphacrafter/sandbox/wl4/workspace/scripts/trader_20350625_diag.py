from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data
import pandas as pd

acct = get_account_dict()
print("=== ACCOUNT ===")
print("total_assets:", acct.get("total_assets"))
print("net_assets:", acct.get("net_assets"))
print("available_cash:", acct.get("available_cash"))
print("market_value:", acct.get("market_value"))
print("gross_position_rate:", acct.get("gross_position_rate"))
print("watch_list:", acct.get("watch_list"))
print("orders:", len(acct.get("orders", [])))
print("--- POSITIONS ---")
for p in acct.get("positions", []):
    print(f"  {p['symbol']:10s} qty={p.get('quantity'):>14.6f} mv={p.get('market_value',0):>14.2f} px={p.get('current_price',0):>12.6f} cost={p.get('cost_price',0):>12.6f} pnl={p.get('profit_loss',0):>12.2f} pnl%={p.get('profit_loss_rate',0)*100:>8.3f}")


def loader(a):
    try:
        return get_stock_daily_data(a, days=120)
    except Exception:
        try:
            return get_index_daily_data(a, days=120)
        except Exception:
            return None


print("\n=== BLOCK RETURNS (window around 06-11 -> last) ===")
assets = acct.get("watch_list", [])
rows = []
for a in assets:
    df = loader(a)
    if df is None or len(df) < 5:
        print(f"  {a}: NO DATA")
        continue
    df = df.sort_values("date").reset_index(drop=True)
    last = df.iloc[-1]
    start_mask = df["date"] >= pd.Timestamp("2025-06-11")
    if start_mask.any():
        idx = start_mask.idxmax()
    else:
        idx = 0
    if idx > 0:
        start_close = df.iloc[idx - 1]["close"]
    else:
        start_close = df.iloc[0]["close"]
    end_close = last["close"]
    ret = end_close / start_close - 1.0
    rows.append((a, float(ret), str(last["date"].date()), float(start_close), float(end_close)))
    print(f"  {a:10s} ret={ret*100:>8.3f}%  ({df.iloc[max(0,idx-1)]['date'].date()}->{last['date'].date()})")

print("\n=== LAST 6 BLOCK RETURNS per asset (10d blocks back from last) ===")
for a in assets:
    df = loader(a)
    if df is None or len(df) < 30:
        continue
    df = df.sort_values("date").reset_index(drop=True)
    closes = df["close"].astype(float)
    blocks = []
    for k in range(0, 60, 10):
        if len(closes) > k + 10:
            r = closes.iloc[-1 - k] / closes.iloc[-11 - k] - 1.0
            blocks.append(round(float(r) * 100, 2))
    print(f"  {a:10s} 10d-block rets (oldest->latest): {blocks}")
