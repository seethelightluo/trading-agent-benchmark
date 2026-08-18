from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acct = get_account_dict()
print("TOTAL_ASSETS", acct.get("total_assets"))
print("NET_ASSETS", acct.get("net_assets"))
print("CASH", acct.get("available_cash"))
print("GROSS_POS", acct.get("gross_position_rate"))
print("N_POSITIONS", len(acct.get("positions", [])))
print("N_ORDERS", len(acct.get("orders", [])))
print("WATCHLIST", acct.get("watch_list"))

pos = {p["symbol"]: p for p in acct.get("positions", [])}
assets = acct.get("watch_list", [])
total = acct.get("net_assets", 0)

print("\n--- per-asset block stats (last ~12 days) ---")
rows = []
for a in assets:
    df = get_stock_daily_data(a, days=15)
    if df is None or len(df) < 3:
        df = get_index_daily_data(a, days=15)
    if df is None or len(df) < 3:
        print(a, "NO DATA")
        continue
    df = df.sort_values("date")
    px0 = float(df.iloc[-12]["close"])  # ~block start
    px1 = float(df.iloc[-1]["close"])
    chg = px1 / px0 - 1.0
    p = pos.get(a)
    mv = p["market_value"] if p else 0.0
    contrib = mv / total * chg
    rows.append((a, chg, contrib, px0, px1, mv))
    print(f"{a:8s} chg {chg*100:7.2f}%  contrib {contrib*100:6.2f}pp  mv {mv:12,.2f}  px {px0:.2f}->{px1:.2f}")
print("\nSUM contrib (pp):", sum(r[2] for r in rows) * 100)
