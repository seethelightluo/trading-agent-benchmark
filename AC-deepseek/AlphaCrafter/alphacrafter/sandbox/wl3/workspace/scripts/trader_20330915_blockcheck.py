"""Trader block check 2033-09-01 -> 2033-09-15: account + per-asset block returns."""
import json
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acct = get_account_dict()
nav = float(acct.get("net_assets", 0.0))
print(f"NAV={nav:,.2f}  total_pnl={acct.get('total_profit_loss',0):,.2f} "
      f"rate={acct.get('total_profit_loss_rate',0)*100:.2f}%")
print(f"gross_pos_rate={acct.get('gross_position_rate',0)*100:.1f}%  cash_avail={acct.get('available_cash',0):,.2f}")
pos = {p["symbol"]: p for p in acct.get("positions", [])}
print(f"n_positions={len(pos)}")

# per-asset block return: close[-11] -> close[-1] (10-day block return)
watch = acct.get("watch_list", [])
for a in watch:
    try:
        df = get_stock_daily_data(a, days=30) if a not in ("DXY","VIX","USDCNY","USDJPY","EURUSD") else get_index_daily_data(a, days=30)
        if df is None or len(df) < 12:
            print(f"{a}: no data"); continue
        c = df["close"].astype(float)
        r = (c.iloc[-1] / c.iloc[-11] - 1.0) * 100
        w = pos.get(a, {}).get("market_value", 0.0) / nav * 100 if nav > 0 else 0.0
        print(f"{a}: block_ret={r:+.2f}%  weight_now={w:.2f}%  qty={pos.get(a,{}).get('quantity',0):.4f}")
    except Exception as e:
        print(f"{a}: err {e}")
