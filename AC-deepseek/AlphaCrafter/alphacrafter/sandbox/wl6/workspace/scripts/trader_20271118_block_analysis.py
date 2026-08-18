"""Analyze new block 2027-11-04 -> 2027-11-18 performance for memory log."""
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
import pandas as pd

acc = get_account_dict()
print("=== ACCOUNT ===")
print("total_assets", round(acc.get("total_assets", 0), 2))
print("net_assets", round(acc.get("net_assets", 0), 2))
print("available_cash", round(acc.get("available_cash", 0), 2))
print("pending orders:", len(acc.get("orders", [])))

pos = {p["symbol"]: p for p in acc.get("positions", [])}
print("\n=== POSITIONS ===")
for s, p in sorted(pos.items(), key=lambda x: -x[1].get("market_value", 0)):
    print(f"{s}: qty={p.get('quantity',0):.4f} mv={p.get('market_value',0):.2f} "
          f"pnl_rate={p.get('profit_loss_rate',0)*100:.2f}%")

print("\n=== PER-ASSET BLOCK RETURNS (2027-11-04 -> 2027-11-18) ===")
for s in acc.get("watch_list", []):
    df = get_stock_daily_data(symbol=s, days=40)
    if df is None or len(df) < 2:
        print(s, "no data")
        continue
    df = df.sort_values("date").reset_index(drop=True)
    target = pd.Timestamp("2027-11-04")
    idx = df.index[df["date"] <= target]
    if len(idx) == 0:
        print(s, "no bar before block start")
        continue
    i0 = idx[-1]
    c0 = df.iloc[i0]["close"]
    c1 = df.iloc[-1]["close"]
    ret = (c1 / c0 - 1.0) * 100
    w = pos.get(s, {}).get("market_value", 0) / acc.get("net_assets", 1)
    print(f"{s}: ret={ret:+.2f}% weight_now={w*100:.2f}%")

# Regime at block end
frames = {}
for a in acc.get("watch_list", []):
    try:
        df = get_stock_daily_data(a, days=60)
        if df is not None and len(df) >= 30:
            frames[a] = df.set_index("date")["close"].astype(float)
    except Exception:
        pass
panel = pd.concat(frames, axis=1, join="inner")
rets = panel.pct_change().dropna()
mkt = rets.mean(axis=1)
r20 = float(mkt.tail(20).mean())
v20 = float(mkt.tail(20).std())
trend = r20 / v20 * (20.0 ** 0.5) if v20 and v20 > 1e-12 else 0.0
regime = "bull" if trend > 1.0 else ("bear" if trend < -1.0 else "sideways")
print("\nregime_at_block_end", regime, "trend_tstat", round(trend, 3))
