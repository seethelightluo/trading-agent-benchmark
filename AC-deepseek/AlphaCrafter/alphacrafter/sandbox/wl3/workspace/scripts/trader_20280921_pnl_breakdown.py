"""Compute approx per-asset PnL contributions for cycle 09-07 -> 09-21 (2028)."""
import json
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

OBS = {"DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"}
acct = json.load(open("../persistent/account.json"))
assets = acct["watch_list"] if "watch_list" in acct else None
# fall back to account positions
positions = {p["symbol"]: p for p in acct.get("positions", [])}
assets = list(positions.keys())

def get_df(sym, days=30):
    try:
        if sym in OBS:
            return get_index_daily_data(sym, days=days)
        return get_stock_daily_data(sym, days=days)
    except Exception:
        return None

# capture the pre-step portfolio weights (drifted 08-10 posture) approx from
# previous account snapshot: XAU 25.0 COPPER 16.8 WTI 13.3 ETH 8.9 SPX 6.9
# 000300 6.5 NDX 5.8 N225 5.4 000688 4.8 SOX 1.8 frozen 0.96 each
w0 = {"XAU": 0.250, "COPPER": 0.168, "WTI": 0.133, "ETH": 0.089, "SPX": 0.069,
      "000300.SH": 0.065, "NDX": 0.058, "N225": 0.054, "000688.SH": 0.048,
      "SOX": 0.018, "HSI": 0.0096, "SX5E": 0.0096, "BTC": 0.0096,
      "US10Y": 0.0096, "CN10Y": 0.0096}
tot0 = sum(w0.values())
w0 = {k: v / tot0 for k, v in w0.items()}

ret = {}
for a in assets:
    df = get_df(a, days=25)
    if df is None or len(df) < 12:
        ret[a] = None
        continue
    df = df.sort_values("date")
    # block: 2028-09-07 open -> 2028-09-20 close approx using closes
    c = df["close"].astype(float)
    ret[a] = c.iloc[-1] / c.iloc[-12] - 1.0  # ~10 trading days

contrib = {}
for a in assets:
    if ret[a] is not None:
        contrib[a] = w0.get(a, 0.0) * ret[a]

nav0 = 838751.26
est_pnl = sum(contrib.values()) * nav0
print(f"estimated block pnl (approx): {est_pnl:,.0f}  (actual +19,946)")
for a in sorted(contrib, key=lambda x: -abs(contrib[x])):
    print(f"  {a:10s} ret={ret[a]*100:7.2f}%  w0={w0.get(a,0)*100:5.2f}%  contrib={contrib[a]*100:6.2f}% (${contrib[a]*nav0:,.0f})")
