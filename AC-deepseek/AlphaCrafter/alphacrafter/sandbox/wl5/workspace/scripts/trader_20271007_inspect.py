"""Trader inspection: account state + per-asset PnL attribution after 2027-10-07->2027-10-21 block."""
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acct = get_account_dict()
print("total_assets:", round(acct.get("total_assets", 0), 2))
print("net_assets:", round(acct.get("net_assets", 0), 2))
print("available_cash:", round(acct.get("available_cash", 0), 4))
print("market_value:", round(acct.get("market_value", 0), 2))
print("gross_position_rate:", acct.get("gross_position_rate"))
print("net_position_rate:", acct.get("net_position_rate"))
print("watch_list:", acct.get("watch_list", []))
print("pending orders:", acct.get("orders", []))

positions = acct.get("positions", [])
print("\npositions:")
tot = 0.0
for p in sorted(positions, key=lambda x: x.get("market_value", 0), reverse=True):
    sym = p["symbol"]
    qty = p.get("quantity", 0)
    mv = p.get("market_value", 0)
    pl = p.get("profit_loss", 0)
    plr = p.get("profit_loss_rate", 0)
    cost = p.get("cost_price", 0)
    px = p.get("current_price", 0)
    w = mv / acct.get("total_assets", 1) if acct.get("total_assets") else 0
    tot += w
    print(f"  {sym:10s} qty={qty:12.4f} cost={cost:10.4f} px={px:10.4f} mv={mv:12.2f} "
          f"w={w*100:6.2f}% pl={pl:12.2f} plr={plr*100:7.2f}%")
print("sum weights:", round(tot, 6))

# Recent 15d returns per asset to see block drivers (10 trading days from ~2027-10-07)
print("\nrecent 15d close series (last 12 closes):")
for sym in acct.get("watch_list", []):
    df = None
    try:
        df = get_stock_daily_data(sym, days=30)
    except Exception:
        df = None
    if df is None or len(df) < 12:
        try:
            df = get_index_daily_data(sym, days=30)
        except Exception:
            df = None
    if df is None or len(df) < 12:
        print(f"  {sym}: no data")
        continue
    closes = df["close"].tolist()
    r10 = (closes[-1] / closes[-11] - 1) * 100 if closes[-11] else 0
    print(f"  {sym:10s} r10={r10:+7.2f}%  closes[-12:]={[round(c,2) for c in closes[-12:]]}")
