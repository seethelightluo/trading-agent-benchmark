from alphacrafter.sim.utils import get_account_dict
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

acc = get_account_dict()
print("total_assets:", acc.get("total_assets"))
print("net_assets:", acc.get("net_assets"))
print("available_cash:", acc.get("available_cash"))
print("market_value:", acc.get("market_value"))
print("gross_position_rate:", acc.get("gross_position_rate"))
print("num_positions:", len(acc.get("positions", [])))
print("orders:", acc.get("orders", []))

pos = {p["symbol"]: p for p in acc.get("positions", [])}
watch = acc.get("watch_list", [])
for a in watch:
    df = get_stock_daily_data(a, days=12)
    if df is None or len(df) < 11:
        df = get_index_daily_data(a, days=12)
    if df is None or len(df) < 11:
        print(a, "no data")
        continue
    df = df.sort_values("date")
    p0 = float(df.iloc[-11]["close"])
    p1 = float(df.iloc[-1]["close"])
    chg = (p1 / p0 - 1) * 100
    q = pos.get(a, {}).get("quantity", 0)
    mv = pos.get(a, {}).get("market_value", 0)
    plr = pos.get(a, {}).get("profit_loss_rate", 0)
    print(f"{a:10s} block_px_chg {chg:7.2f}%  qty {q:12.4f}  mv {mv:12.2f}  plr {plr:7.2f}%")
