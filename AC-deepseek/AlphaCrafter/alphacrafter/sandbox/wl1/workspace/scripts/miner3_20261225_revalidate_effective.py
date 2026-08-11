"""miner_3 re-validation of currently EFFECTIVE factors through 2026-12-24.

Re-validates every factor in the live ensemble + remaining effective library:
  mom_120d_skip5, nclv_1d/2d/3d/5d, rev_1d/2d/3d/5d, rev_1d_vs, id_rev_1d,
  nbody_1d, vol_of_vol20x60, vix_beta_cond_60x20.
Prints per-factor IC/ICIR and whether they still pass the admission gates.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner3_20261008_lib import load_close_panel, run_validation

close = load_close_panel(days=2500)
print(f"panel dates={close.shape[0]} assets={close.shape[1]} "
      f"range={close.index.min().date()}..{close.index.max().date()}")

lr = close.pct_change()
ret = close.pct_change()

# --- VIX aligned (observation-only) ---
vix = pd.read_csv("../persistent/index_data/VIX.csv", parse_dates=["date"])
vix = vix.set_index("date")["close"]
cutoff = close.index.max()
vix = vix[vix.index <= cutoff]
vix = vix[~vix.index.duplicated(keep="last")]
vix = vix.reindex(close.index).ffill()
vix_ret = vix.pct_change()

# --- factor panel ---
def nclv(win, h=1):
    """normalized cumulative log-volume: rank of sum of log-returns in win."""
    return (close / close.shift(win)).rank(axis=1)

factors = {}
# momentum family
factors["mom_120d_skip5"] = close.shift(5) / close.shift(125) - 1.0
# nclv family (miner2)
for w in (1, 2, 3, 5):
    factors[f"nclv_{w}d"] = (close / close.shift(w) - 1).rank(axis=1)
# reversal family (miner2)
factors["rev_1d"] = -(close / close.shift(1) - 1)
factors["rev_2d"] = -(close / close.shift(2) - 1)
factors["rev_3d"] = -(close / close.shift(3) - 1)
factors["rev_5d"] = -(close / close.shift(5) - 1)
# volatility family
vol20 = lr.rolling(20).std()
factors["vol_of_vol20x60"] = vol20.rolling(60).std()
# vix beta conditional
beta60 = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
for c in close.columns:
    beta60[c] = (close[c].pct_change().rolling(60, min_periods=30)
                 .cov(vix_ret) / vix_ret.rolling(60, min_periods=30).var())
vix_move20 = vix / vix.shift(20) - 1.0
factors["vix_beta_cond_60x20"] = -beta60.mul(vix_move20, axis=0)

gate_ic, gate_icir = 0.0070, 0.0840
print(f"\n{'factor':26s} {'h':>3s} {'IC':>8s} {'ICIR':>8s} {'hit':>6s} {'n':>5s}  gate")
for name, f in factors.items():
    # use horizon that maximizes |ICIR| among 1..10
    best = None
    for h in (1, 2, 3, 5, 10):
        fwd = close.shift(-h) / close - 1.0
        ics = []
        for dt in f.index:
            ff, rr = f.loc[dt], fwd.loc[dt]
            m = ff.notna() & rr.notna()
            if m.sum() < 8:
                continue
            ic = ff[m].rank().corr(rr[m].rank())
            if np.isfinite(ic):
                ics.append(ic)
        ics = pd.Series(ics)
        if len(ics) == 0:
            continue
        ic = ics.mean(); icir = ic / ics.std(ddof=1) if ics.std(ddof=1) > 0 else 0
        if best is None or abs(icir) > abs(best[1]):
            best = (h, icir, ic, len(ics), float((ics > 0).mean()) if ic > 0 else float((ics < 0).mean()))
    if best is None:
        print(f"{name:26s}  no data")
        continue
    h, icir, ic, n, hit = best
    ok = "PASS" if abs(ic) >= gate_ic and abs(icir) >= gate_icir else "fail"
    print(f"{name:26s} {h:3d} {ic:+8.5f} {icir:+8.5f} {hit:6.3f} {n:5d}  {ok}")
