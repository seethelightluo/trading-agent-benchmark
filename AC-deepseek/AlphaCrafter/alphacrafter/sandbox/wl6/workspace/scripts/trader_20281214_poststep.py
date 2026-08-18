"""Trader post-step analysis: account state after the 2028-12-14 block and
block attribution vs the previous block snapshot (read-only)."""
import json
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

PREV = {
    "000300.SH": (7.2918, 4424.8820), "SPX": (19.7096, 6832.8469),
    "HSI": (0.2244, 25008.5996), "N225": (2.6671, 50494.2660),
    "SX5E": (5.4964, 5870.2854), "000688.SH": (3.0383, 1846.8770),
    "SOX": (13.4850, 9986.8700), "NDX": (1.7687, 18242.4978),
    "XAU": (28.1804, 4778.9537), "COPPER": (19448.7772, 6.9245),
    "WTI": (408.2937, 79.0250), "BTC": (0.2765, 299389.5787),
    "ETH": (28.6122, 2990.6691), "US10Y": (20319.5002, 6.6278),
    "CN10Y": (3218.0833, 1.7437),
}
acc = get_account_dict()
print("=== ACCOUNT (post-block) ===")
print("net_assets:", round(acc["net_assets"], 2), " cash:", acc["available_cash"],
      " gross_pos:", acc["gross_position_rate"])
pnl_total = 0.0
print(f"\n{'sym':>10} {'qty':>12} {'px':>12} {'mv':>14} {'prev_px':>12} {'ret':>9} {'est_pnl':>12}")
for p in acc.get("positions", []):
    s = p["symbol"]
    q = p["quantity"]; px = p["current_price"]; mv = p["market_value"]
    prev_q, prev_px = PREV.get(s, (0.0, None))
    ret = px / prev_px - 1 if prev_px else 0.0
    # PnL approx: current MV - prev MV (same qty => price move; if qty changed, note it)
    prev_mv = prev_q * prev_px if prev_px else 0.0
    pnl = mv - prev_mv
    pnl_total += pnl
    flag = "" if abs(q - prev_q) < 1e-6 else f"  <qty changed {prev_q:.2f}->{q:.2f}>"
    print(f"{s:>10} {q:>12.4f} {px:>12.4f} {mv:>14.2f} {prev_px:>12.4f} {ret:>9.2%} {pnl:>12.2f}{flag}")
print(f"\nSum est PnL: {pnl_total:,.2f}  |  actual net chg: {acc['net_assets'] - 1090345.58:,.2f}")

# current date data for next-cycle reference
for a in acc.get("watch_list", []):
    df = get_stock_daily_data(a, days=30)
    if df is None or len(df) == 0:
        continue
    c = df["close"].astype(float)
    print(f"{a:>10} last={c.iloc[-1]:>12.4f} d={df['date'].iloc[-1]} ret5={c.iloc[-1]/c.iloc[-6]-1:>8.2%}")
