"""Trader cycle review 2032-02-06..2032-02-19: inspect account, per-asset PnL."""
import json
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acct = get_account_dict()
print("total_assets:", acct.get("total_assets"))
print("net_assets:", acct.get("net_assets"))
print("available_cash:", acct.get("available_cash"))
print("gross_position_rate:", acct.get("gross_position_rate"))
print("n_positions:", len(acct.get("positions", [])))
print("n_orders:", len(acct.get("orders", [])))

OBS = {"DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"}
positions = {p["symbol"]: p for p in acct.get("positions", [])}
watch = acct.get("watch_list", [])

tot_nav = float(acct.get("net_assets", 0.0))
contrib = []
for sym in watch:
    pos = positions.get(sym)
    if pos is None:
        contrib.append((sym, 0.0, 0.0, 0.0, "NO_POS"))
        continue
    mv = float(pos.get("market_value", 0.0))
    qty = float(pos.get("quantity", 0.0))
    cost = float(pos.get("cost_price", 0.0))
    cur = float(pos.get("current_price", 0.0))
    pl = float(pos.get("profit_loss", 0.0))
    w = mv / tot_nav if tot_nav else 0.0
    ret = (cur / cost - 1.0) if cost else 0.0
    contrib.append((sym, w, ret, pl, f"{qty:.4f}@{cost:.4f}->{cur:.4f}"))

contrib.sort(key=lambda x: -x[3])
print("\n--- positions sorted by PnL ---")
for sym, w, ret, pl, info in contrib:
    print(f"{sym:10s} w={w*100:6.2f}% ret={ret*100:7.2f}% pnl={pl:10.2f} {info}")

# block return attribution from cost basis (rebalance was at block start)
print("\n--- weight * (cur/cost - 1) attribution ---")
for sym, w, ret, pl, info in contrib:
    print(f"{sym:10s} w*ret={w*ret*100:7.3f}%")
print(f"sum contrib ~ {sum(w*ret for _, w, ret, _, _ in contrib)*100:.3f}%")

# try to get raw price data for block returns 02-05..02-19
print("\n--- raw close data (last 3 rows) ---")
for sym in watch:
    try:
        df = get_stock_daily_data(symbol=sym, days=30) if sym not in OBS else get_index_daily_data(symbol=sym, days=30)
        if df is None or len(df) < 3:
            print(f"{sym:10s} NA")
            continue
        closes = df["close"].astype(float).tolist()
        dates = [str(d)[:10] for d in df["date"]]
        blk = (closes[-1] / closes[-12] - 1.0) if len(closes) >= 12 else float("nan")
        print(f"{sym:10s} block_ret={blk*100:8.2f}%  last={closes[-1]:.2f} ({dates[-1]})")
    except Exception as e:
        print(f"{sym:10s} ERR {e}")
