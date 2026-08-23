"""miner_1 2030-12-12: re-validate the two currently-effective factors
(flip_mom_20x10, usdcny_beta_60) through visible_through for recency/sign-drift."""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from miner_1_common import (load_panel, load_macro_panel, forward_returns,
                            spearman_ic_series, ic_metrics, regime_slices,
                            decay_by_horizon, coverage, visible_through)

px, vol = load_panel(start="2020-01-01")
print("asof:", visible_through().date(), "panel:", px.shape)

def flip_mom_20x10(panel):
    c20 = panel.shift(20); c10 = panel.shift(10)
    m20 = panel/c20 - 1.0
    s10 = np.sign(panel/c10 - 1.0)
    return s10 * m20

def usdcny_beta_60(panel, mac):
    rets = panel.pct_change()
    r60 = rets.rolling(60)
    cov = r60.cov(mac.shift(1).pct_change())
    var = mac.shift(1).pct_change().rolling(60).var()
    b = cov / var
    return b

mac = load_macro_panel("USDCNY", start="2020-01-01")

for name, fac in [("flip_mom_20x10", flip_mom_20x10(px)),
                  ("usdcny_beta_60", usdcny_beta_60(px, mac))]:
    print("\n" + "="*70)
    print("FACTOR:", name)
    fwd10 = forward_returns(px, horizon=10)
    ics = spearman_ic_series(fac, fwd10)
    st = ic_metrics(ics)
    print("full-sample H10 IC metrics:", {k: (round(v,4) if isinstance(v,float) else v) for k,v in st.items()})
    for w,lab in [(126,"recent~6m"),(252,"recent~12m"),(504,"recent~24m"),(756,"recent~36m")]:
        sub = ics.iloc[-w:]
        m = ic_metrics(sub)
        print(f"  {lab}: IC={m['ic']:.4f} ICIR={m['icir']:.4f} hit={m['hit']:.3f} n={m['n_ic_dates']}")
    print("  decay H:", decay_by_horizon(px, fac))
    print("  coverage_asset_days:", round(coverage(fac, px),4))
    try:
        print("  regime:", regime_slices(ics))
    except Exception as e:
        print("  regime err", e)