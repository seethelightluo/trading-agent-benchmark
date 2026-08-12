import json
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acct = get_account_dict()
print("total_assets:", round(acct.get("total_assets", 0), 2))
print("net_assets:", round(acct.get("net_assets", 0), 2))
print("available_cash:", round(acct.get("available_cash", 0), 2))
print("gross_position_rate:", acct.get("gross_position_rate"))
print("net_position_rate:", acct.get("net_position_rate"))
print("positions:", len(acct.get("positions", [])))
print("pending orders:", len(acct.get("orders", [])))

def get_data(sym):
    df = get_stock_daily_data(sym, days=20)
    if df is None or len(df) == 0:
        df = get_index_daily_data(sym, days=20)
    return df

rows = []
for p in acct.get("positions", []):
    sym = p["symbol"]
    qty = p.get("quantity", 0)
    cost = p.get("cost_price", 0)
    cur = p.get("current_price", 0)
    mv = p.get("market_value", 0)
    pnl = p.get("profit_loss", 0)
    pnlr = p.get("profit_loss_rate", 0)
    df = get_data(sym)
    blk_ret = None
    if df is not None and len(df) >= 11:
        c = df["close"].astype(float)
        blk_ret = (c.iloc[-1] / c.iloc[-11] - 1) * 100
    rows.append((sym, qty, cost, cur, mv, pnl, pnlr, blk_ret))

rows.sort(key=lambda r: r[5], reverse=True)
print("\n{:<10} {:>10} {:>12} {:>12} {:>14} {:>12} {:>8} {:>8}".format(
    "SYM", "QTY", "COST", "CUR", "MV", "PNL", "PNL%", "BLK10%"))
for r in rows:
    print("{:<10} {:>10.4f} {:>12.4f} {:>12.4f} {:>14.2f} {:>12.2f} {:>7.2f}% {:>7.2f}%".format(*r))
wsum = sum(p.get("market_value", 0) for p in acct.get("positions", []))
print("\nweights sum (mv/net):", round(wsum / acct.get("net_assets", 1), 4))
