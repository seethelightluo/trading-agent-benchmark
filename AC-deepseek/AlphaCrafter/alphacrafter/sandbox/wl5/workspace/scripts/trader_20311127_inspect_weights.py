"""Inspect current (2031-11-26 visible) composite scores and target weights."""
import sys, os
sys.path.insert(0, os.getcwd())
import strategy as st
import numpy as np
from alphacrafter.sim.utils import get_account_dict

acc = get_account_dict()
assets = list(acc["watch_list"])
print("watch_list:", assets)

ensemble = st._load_ensemble()
print("ensemble (id, w, dir):")
for fid, w, d in ensemble:
    print(f"  {fid:24s} {w:.4f} {d:+d}")

closes = st._closes(assets)
panel = st.pd.DataFrame(closes).sort_index()
rets = panel.pct_change()

dxy_c = st._macro_close("DXY"); dxy_r = dxy_c.pct_change() if dxy_c is not None else None
vix_c = st._macro_close("VIX"); vix_r = vix_c.pct_change() if vix_c is not None else None
cny_c = st._macro_close("USDCNY"); cny_r = cny_c.pct_change() if cny_c is not None else None

fvals = {fid: {} for fid, _, _ in ensemble}
for a in assets:
    c = closes.get(a); r = rets[a] if a in rets else None
    if c is None or r is None:
        continue
    for fid, _, _ in ensemble:
        try:
            if fid == "trend_r2_30_signed": v = st._trend_r2(c)
            elif fid == "semi_down_ratio_20": v = st._semi_down_ratio(r)
            elif fid == "mom_120d_skip5": v = st._mom_120(c)
            elif fid == "mom_10d_skip5": v = st._mom_10(c)
            elif fid == "vol_of_vol20x60": v = st._vol_of_vol(r)
            elif fid == "time_under_water_120": v = st._underwater(c)
            elif fid == "tail_ratio_20": v = st._tail_ratio(r)
            elif fid == "dxy_beta_60": v = st._beta_60(r, dxy_r) if dxy_r is not None else None
            elif fid == "cny_beta_60": v = st._beta_60(r, cny_r) if cny_r is not None else None
            elif fid == "vix_beta_cond_60x20": v = st._vix_beta_cond(r, vix_r, vix_c) if vix_r is not None else None
            else: v = None
        except Exception:
            v = None
        fvals[fid][a] = v

score = {a: 0.0 for a in assets}
for fid, w, direction in ensemble:
    rk = st._rank_map(fvals[fid], assets)
    for a in assets:
        score[a] += w * direction * rk[a]

print("\ncomposite scores:")
for a in sorted(assets, key=lambda x: -score[x]):
    print(f"  {a:10s} {score[a]:+.4f}")

sv = np.array([score[a] for a in assets])
print("score min/max/std:", sv.min().round(4), sv.max().round(4), sv.std().round(4))

market = rets.mean(axis=1)
trend20 = float(market.tail(20).mean()) if len(market) >= 20 else 0.0
avg_px = float(panel.mean(axis=1).iloc[-1]) if len(panel) else 0.0
ma60 = float(panel.mean(axis=1).tail(60).mean()) if len(panel) >= 60 else avg_px
bearish = trend20 < 0.0 and avg_px < ma60
print("trend20:", round(trend20, 5), "bearish:", bearish)

regime_w = {}
for a in assets:
    if bearish and a in st.DEF: regime_w[a] = 2.2
    elif bearish: regime_w[a] = 0.75
    else: regime_w[a] = 1.0

wts = st._to_weights(score, assets, regime_w)
print("\ntarget weights:")
for a in sorted(assets, key=lambda x: -wts[x]):
    print(f"  {a:10s} {wts[a]*100:6.2f}%")
print("sum:", round(sum(wts.values()), 6))
