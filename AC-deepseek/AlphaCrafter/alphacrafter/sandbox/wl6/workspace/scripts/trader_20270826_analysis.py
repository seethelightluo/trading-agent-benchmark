"""Post-block analysis: 2027-08-26 -> 2027-09-09 block drivers."""
import json
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]


def loader(a, n=60):
    try:
        return get_stock_daily_data(a, days=n)
    except Exception:
        return None


def idx_loader(a, n=60):
    try:
        return get_index_daily_data(a, days=n)
    except Exception:
        return None


def get_frame(a):
    f = loader(a)
    if f is None or len(f) == 0:
        f = idx_loader(a)
    return f


# Pre-step (block start 2027-08-26) holdings from account.json.bak
bak = json.load(open("../persistent/account.json.bak"))
pre = {p["symbol"]: p for p in bak["positions"]}
cur = json.load(open("../persistent/account.json"))
post = {p["symbol"]: p for p in cur["positions"]}

print(f"pre net_assets={bak['net_assets']:.2f}  post net_assets={cur['net_assets']:.2f}  "
      f"block_pnl={cur['net_assets']-bak['net_assets']:+.2f}")

rows = []
for a in ASSETS:
    f = get_frame(a)
    if f is None or len(f) < 25:
        rows.append((a, None, None, None, None, None, None, None))
        continue
    f = f.sort_values("date").reset_index(drop=True)
    px_start = float(f.iloc[-11]["close"])   # close on 2027-08-25 (last completed before decision)
    px_end = float(f.iloc[-1]["close"])      # close 2027-09-09
    ret = px_end / px_start - 1.0
    q0 = pre.get(a, {}).get("quantity", 0.0)
    q1 = post.get(a, {}).get("quantity", 0.0)
    mv0 = q0 * px_start
    mv1 = q1 * px_end
    dq = q1 - q0
    rows.append((a, ret, q0, q1, dq, mv0, mv1, mv1 - mv0))

print(f"\n{'asset':9s} {'ret':>8s} {'qty0':>11s} {'qty1':>11s} {'dqty':>10s} {'mv0':>10s} {'mv1':>10s} {'dMv':>10s}")
for a, ret, q0, q1, dq, mv0, mv1, dmv in rows:
    if ret is None:
        print(f"{a:9s}  n/a")
        continue
    print(f"{a:9s} {ret:8.2%} {q0:11.2f} {q1:11.2f} {dq:10.2f} {mv0:10.0f} {mv1:10.0f} {dmv:10.0f}")

# weight vector post-block
tot = cur["net_assets"]
print("\npost-block weights:")
for a in ASSETS:
    p = post.get(a)
    if p:
        w = p["market_value"] / tot
        print(f"  {a:9s} w={w:6.2%}  pnl_rate={p.get('profit_loss_rate', 0):8.2%}  pnl={p.get('profit_loss', 0):10.1f}")

# regime check at block end
frames = {}
for a in ASSETS:
    f = get_frame(a)
    if f is not None and len(f) >= 30:
        frames[a] = f.set_index("date")["close"].rename(a)
panel = pd.concat(frames, axis=1, join="inner").dropna()
rets = panel.pct_change().dropna()
mkt = rets.mean(axis=1)
r20 = float(mkt.tail(20).mean())
v20 = float(mkt.tail(20).std())
trend = r20 / v20 * (20 ** 0.5) if v20 and v20 > 1e-12 else 0.0
regime = "bull" if trend > 1.0 else ("bear" if trend < -1.0 else "sideways")
print(f"\nregime at block end: {regime} (trend t-stat={trend:.2f}, 20d drift={r20:.4%})")
