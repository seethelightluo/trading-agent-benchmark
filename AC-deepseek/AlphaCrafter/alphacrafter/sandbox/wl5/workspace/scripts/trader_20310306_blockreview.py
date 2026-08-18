"""Trader block review: 2031-02-20 -> 2031-03-06. Read-only account + price analysis."""
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

acct = get_account_dict()
print("total_assets:", round(acct.get("total_assets", 0), 2))
print("available_cash:", round(acct.get("available_cash", 0), 2))
print("gross_position_rate:", acct.get("gross_position_rate"))
print("net_position_rate:", acct.get("net_position_rate"))
print("n orders:", len(acct.get("orders", [])))

positions = {p["symbol"]: p for p in acct.get("positions", [])}
print("\npositions:")
for sym, p in sorted(positions.items()):
    print(f"  {sym}: qty={p.get('quantity'):.4f} mv={p.get('market_value',0):.2f} plr={p.get('profit_loss_rate',0)*100:.2f}%")

watch = acct.get("watch_list", [])
print("\nwatch_list:", watch)

print("\nblock drivers (02-19 close -> latest close):")
total_mv = sum(p.get("market_value", 0) for p in positions.values())
for sym in watch:
    df = get_stock_daily_data(symbol=sym, days=25)
    if df is None or len(df) < 12:
        print(f"  {sym}: no data")
        continue
    df = df.sort_values("date")
    px_prev = df.iloc[-11]["close"]   # close ~10 trading days ago (block start px, 02-19)
    px_now = df.iloc[-1]["close"]
    ret = (px_now / px_prev - 1) * 100
    pos = positions.get(sym)
    mv = pos.get("market_value", 0) if pos else 0.0
    contrib = mv / total_mv * ret if total_mv else 0.0
    print(f"  {sym}: {ret:+.2f}%  mv={mv:12.2f}  contrib={contrib:+.2f}pp")
