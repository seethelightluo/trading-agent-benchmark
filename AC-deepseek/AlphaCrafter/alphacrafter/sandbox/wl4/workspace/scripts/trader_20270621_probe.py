"""Trader probe 2027-06-21: ensemble sync check, regime check (VIX, SPX vs 20d MA), account state."""
import json
from pathlib import Path
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data

# 1) ensemble files
root_ens = json.loads(Path("factor_ensemble.json").read_text())
lib_ens = json.loads(Path("factors/factor_ensemble.json").read_text())
print("ROOT ensemble:", [(f["factor_id"], f["weight"], f["direction"]) for f in root_ens["selected_factors"]])
print("LIB  ensemble:", [(f["factor_id"], f["weight"], f["direction"]) for f in lib_ens["selected_factors"]])

# 2) regime check (SPX is a tradable benchmark -> stock data path)
spx = get_stock_daily_data("SPX", days=40)
if spx is not None and len(spx) >= 25:
    spx = spx.sort_values("date")
    close = spx["close"].astype(float)
    last = float(close.iloc[-1])
    ma20 = float(close.tail(21).iloc[:-1].mean())  # 20d MA through prev close
    print(f"SPX last={last:.2f} 20dMA(prev)={ma20:.2f} above={last > ma20}")
vix = get_index_daily_data("VIX", days=30)
if vix is not None and len(vix) > 0:
    vix = vix.sort_values("date")
    print(f"VIX last={float(vix['close'].iloc[-1]):.2f} (prev 5: {[round(float(x),2) for x in vix['close'].iloc[-6:-1]]})")

# 3) account
acc = get_account_dict()
print("net_assets:", acc.get("net_assets"), "total:", acc.get("total_assets"),
      "cash:", acc.get("available_cash"), "gross_pos_rate:", acc.get("gross_position_rate"))
print("positions:", [(p["symbol"], round(p.get("quantity", 0), 4), round(p.get("market_value", 0), 2)) for p in acc.get("positions", [])])
print("orders:", acc.get("orders", []))
print("watch_list:", acc.get("watch_list", []))
