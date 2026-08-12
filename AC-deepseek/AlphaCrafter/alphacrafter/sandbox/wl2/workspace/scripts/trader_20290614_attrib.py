import json
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

ASSETS = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU",
          "COPPER","WTI","BTC","ETH","US10Y","CN10Y"]

acc = get_account_dict()
na = acc["net_assets"]
pos = {p["symbol"]: p for p in acc.get("positions", [])}
tot_mv = sum(p["market_value"] for p in pos.values()) or 1.0
w = {a: pos[a]["market_value"] / tot_mv for a in ASSETS if a in pos}

# block window: 2029-05-31 -> 2029-06-14 (10 trading days)
rows = []
for a in ASSETS:
    df = get_stock_daily_data(a, days=30)
    if df is None or len(df) < 12:
        rows.append((a, float("nan"), w.get(a, 0.0)))
        continue
    d = df["date"].astype(str).tolist()
    c = df["close"].astype(float).tolist()
    # find close on/just before 2029-05-31 and last close
    start = None
    for i, dd in enumerate(d):
        if dd <= "2029-05-31":
            start = c[i]
        else:
            break
    end = c[-1]
    r = (end / start - 1.0) * 100 if start and start > 0 else float("nan")
    rows.append((a, r, w.get(a, 0.0)))

tot_attr = 0.0
print(f"{'asset':>10s} {'blk_ret%':>9s} {'wt':>7s} {'attr_bp':>8s}")
for a, r, wt in sorted(rows, key=lambda x: -(x[1] if x[1] == x[1] else -999)):
    attr = r * wt if r == r else 0.0
    tot_attr += attr
    print(f"{a:>10s} {r:>9.2f} {wt:>7.4f} {attr:>8.2f}")
print(f"sum attr (approx block pnl bp): {tot_attr:.2f}")
print(f"NAV: {na:.2f}  (block return from step: -1.99%)")
