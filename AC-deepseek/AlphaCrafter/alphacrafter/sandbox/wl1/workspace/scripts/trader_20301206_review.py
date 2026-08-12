"""Trader 2030-12-06 cycle review: reconstruct executed weights + block contribs."""
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

acc = get_account_dict()
assets = acc["watch_list"]
NAV_end = acc["total_assets"]
NAV_start = 948778.30  # from prior block-end memory

# Fetch 170d frames; find close on 2030-12-05 (recon) and 2030-12-19/20 (block end)
frames = {}
for a in assets:
    df = get_stock_daily_data(symbol=a, days=170)
    if df is None or len(df) < 40:
        frames[a] = None
        continue
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    frames[a] = df

def close_on(a, dt):
    df = frames.get(a)
    if df is None:
        return None
    sub = df[df.index <= pd.Timestamp(dt)]
    if sub.empty:
        return None
    return float(sub["close"].iloc[-1])

pos = {p["symbol"]: p for p in acc["positions"]}

print("=== Executed target (recon @1205 close) ===")
target_val = {}
for a in assets:
    q = pos.get(a, {}).get("quantity", 0)
    px = close_on(a, "2030-12-05")
    if px:
        target_val[a] = q * px
tot = sum(target_val.values())
for a, v in sorted(target_val.items(), key=lambda x: -x[1]):
    print(f"  {a:10s} qty {pos[a]['quantity']:.4f} recon_px {close_on(a,'2030-12-05'):.4f} val {v:9.2f} w {v/tot*100:5.2f}%")
print(f"  target total {tot:.2f} (NAV {NAV_end:.2f})")

print("\n=== Block contribs (recon 1205 -> block end 1219/1220) ===")
end_dt = "2030-12-19"
contribs = {}
for a in assets:
    p0 = close_on(a, "2030-12-05")
    p1 = close_on(a, end_dt)
    if p0 and p1 and a in pos:
        ret = p1 / p0 - 1.0
        contribs[a] = (ret, pos[a]["quantity"] * p0, pos[a]["quantity"] * p0 * ret)
tot_contrib = sum(v[2] for v in contribs.values())
for a, (ret, val, c) in sorted(contribs.items(), key=lambda x: -x[1][2]):
    print(f"  {a:10s} ret {ret*100:7.2f}%  w0 {val/tot*100:5.2f}%  contrib {c:9.2f} ({c/NAV_start*100:6.2f}% NAV)")
print(f"  total contrib {tot_contrib:.2f} ({tot_contrib/NAV_start*100:.2f}%) vs NAV move {(NAV_end-NAV_start)/NAV_start*100:.2f}%")

# Regime at decision (20d mean daily ret through 1205)
print("\n=== Regime @1205 ===")
rets = []
for a in assets:
    df = frames.get(a)
    if df is not None and len(df) >= 25:
        sub = df[df.index <= pd.Timestamp("2030-12-05")]
        if len(sub) >= 25:
            rets.append(float(sub["close"].pct_change().tail(20).mean()))
m = float(np.mean(rets))
above = sum(1 for a in assets if close_on(a, "2030-12-05") and frames[a] is not None and
            close_on(a, "2030-12-05") > float(frames[a]["close"].rolling(20).mean().iloc[-1]))
print(f"  20d mean daily ret {m*100:.4f}%  above_MA20 {above}/15")
for a in assets:
    df = frames.get(a)
    if df is None:
        continue
    sub = df[df.index <= pd.Timestamp("2030-12-05")]
    if len(sub) >= 20:
        c = float(sub["close"].iloc[-1]); ma = float(sub["close"].rolling(20).mean().iloc[-1])
        print(f"  {a:10s} close {c:9.2f} ma20 {ma:9.2f} {'ABOVE' if c>ma else 'below'}")
