"""Trader block inspection: account state + per-asset block returns 2028-06-01 -> 2028-06-15."""
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acc = get_account_dict()
print("=== ACCOUNT ===")
print("total_assets", round(acc.get("total_assets", 0), 2))
print("net_assets", round(acc.get("net_assets", 0), 2))
print("available_cash", round(acc.get("available_cash", 0), 2))
print("market_value", round(acc.get("market_value", 0), 2))
print("gross_position_rate", acc.get("gross_position_rate"))
print("net_position_rate", acc.get("net_position_rate"))
print("watch_list", acc.get("watch_list", []))
print("\n=== POSITIONS ===")
for p in acc.get("positions", []):
    print(f"{p['symbol']:10s} qty={p.get('quantity'):>12.4f} cost={p.get('cost_price'):>10.4f} "
          f"px={p.get('current_price'):>10.4f} mv={p.get('market_value'):>12.2f} "
          f"pl={p.get('profit_loss'):>10.2f} pl_pct={p.get('profit_loss_rate'):>7.2%}")
print("\n=== PENDING ORDERS ===")
for o in acc.get("orders", []):
    print(o)

# Block returns per asset: from close before 06-01 (05-31) to latest close
print("\n=== BLOCK RETURNS (2028-05-31 close -> latest) ===")
for a in acc.get("watch_list", []):
    df = None
    try:
        df = get_stock_daily_data(a, days=20)
    except Exception:
        pass
    if df is None or len(df) < 3:
        try:
            df = get_index_daily_data(a, days=20)
        except Exception:
            pass
    if df is None or len(df) < 3:
        print(f"{a:10s} NO DATA")
        continue
    df = df.sort_values("date")
    c = df["close"].astype(float)
    # find close on/before 2028-05-31 and last close
    dates = df["date"].astype(str)
    pre = c[dates <= "2028-05-31"]
    p0 = float(pre.iloc[-1]) if len(pre) else float(c.iloc[0])
    p1 = float(c.iloc[-1])
    d0 = dates.iloc[len(pre) - 1] if len(pre) else dates.iloc[0]
    d1 = dates.iloc[-1]
    print(f"{a:10s} {d0}->{d1}  {p0:>12.4f} -> {p1:>12.4f}  {(p1/p0-1)*100:>7.2f}%")
