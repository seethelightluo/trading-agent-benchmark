import json
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acc = get_account_dict()
print("total_assets", acc.get("total_assets"))
print("net_assets", acc.get("net_assets"))
print("available_cash", acc.get("available_cash"))
print("market_value", acc.get("market_value"))
print("gross_position_rate", acc.get("gross_position_rate"))
print("orders", acc.get("orders"))
positions = {p["symbol"]: p for p in acc.get("positions", [])}
print("n_positions", len(positions))
for s, p in sorted(positions.items()):
    print(f"{s}: qty={p.get('quantity')} mv={p.get('market_value'):,.2f} pl={p.get('profit_loss'):,.2f} plr={p.get('profit_loss_rate')*100:.2f}%")

watch = acc.get("watch_list", [])
print("watch_list", watch)

def closes(sym, days=15):
    df = None
    try:
        df = get_stock_daily_data(sym, days=days)
    except Exception:
        df = None
    if df is None or len(df) < 2:
        try:
            df = get_index_daily_data(sym, days=days)
        except Exception:
            df = None
    return df

for s in watch:
    df = closes(s, days=20)
    if df is None or len(df) < 2:
        print(s, "no data")
        continue
    first = float(df.iloc[0]["close"])
    last = float(df.iloc[-1]["close"])
    chg = (last / first - 1) * 100
    qty = positions.get(s, {}).get("quantity", 0) if s in positions else 0
    print(f"{s}: {first:.2f} -> {last:.2f} ({chg:+.2f}%) qty={qty}")
