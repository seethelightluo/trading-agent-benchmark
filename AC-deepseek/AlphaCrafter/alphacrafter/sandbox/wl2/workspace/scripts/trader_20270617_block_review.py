"""Trader cycle16 (06-17 -> 07-01) block review: per-asset 10d returns and
approximate contribution using ending market values."""
import json
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

a = json.load(open("../persistent/account.json"))
mv = {p["symbol"]: p["market_value"] for p in a.get("positions", [])}
na = a["net_assets"]

print(f"net_assets={na:.0f} cash={a['available_cash']:.2f}")
rows = []
for s in WATCH:
    try:
        df = get_stock_daily_data(s, days=40)
    except Exception:
        df = None
    if df is None or len(df) < 21:
        print(s, "no data")
        continue
    c = df["close"].astype(float)
    # block start price ~ 10 trading days ago (last-11th close)
    r10 = c.iloc[-1] / c.iloc[-11] - 1.0
    w = mv.get(s, 0.0) / na
    rows.append((s, r10, w, w * r10))
rows.sort(key=lambda x: -x[1])
print(f"{'asset':10s} {'10d ret':>8s} {'end wt':>7s} {'wt*ret(bp)':>10s}")
tot = 0.0
for s, r, w, c_ in rows:
    print(f"{s:10s} {r*100:7.2f}% {w*100:6.2f}% {c_*10000:9.1f}")
    tot += c_
print(f"sum wt*ret approx = {tot*100:.2f}%")
