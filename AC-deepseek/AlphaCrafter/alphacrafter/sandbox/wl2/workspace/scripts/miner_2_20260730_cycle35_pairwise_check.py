"""miner_2 cycle35: pairwise correlation check among passing consistency factors.

Three candidates passed the IC/ICIR admission gate:
  pos_freq_20        IC=0.0450  ICIR=0.1473  maxlibcorr=0.4771
  max_consec_loss_20 IC=-0.0420 ICIR=-0.1432 maxlibcorr=0.3253
  max_consec_gain_20 IC=0.0682  ICIR=0.2310  maxlibcorr=0.3418
They are same-family variants; measure mutual rank rho to decide what to persist
without guaranteed quarantine (screener pairwise |rho| < 0.5 gate).
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "scripts")
from miner2_lib import load_close_panel, per_asset, panel_rank_corr

close = load_close_panel()
idx = close.index


def pos_freq(s, w=20, mp=10):
    def _f(x):
        x = np.asarray(x, dtype=float)
        r = np.diff(x) / x[:-1]
        if len(r) < mp:
            return np.nan
        return float(np.mean(r > 0))
    return s.rolling(w + 1, min_periods=mp + 1).apply(_f, raw=True)


def max_consec(s, w=20, mp=10, direction=1):
    def _m(x):
        x = np.asarray(x, dtype=float)
        r = np.diff(x) / x[:-1]
        if len(r) < mp:
            return np.nan
        best = cur = 0
        for v in r:
            cur = cur + 1 if (v < 0 if direction < 0 else v > 0) else 0
            best = max(best, cur)
        return float(best)
    return s.rolling(w + 1, min_periods=mp + 1).apply(_m, raw=True)


cands = {
    "pos_freq_20": per_asset(close, pos_freq),
    "max_consec_loss_20": per_asset(close, max_consec, 20, 10, -1),
    "max_consec_gain_20": per_asset(close, max_consec, 20, 10, +1),
}

print("pairwise rank rho among candidates:")
names = list(cands)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        rho = panel_rank_corr(cands[names[i]], cands[names[j]])
        print(f"  {names[i]:20s} vs {names[j]:20s} = {rho:+.4f}")

print("\nvs library artifacts:")
EFF = ["mom20_volproxy60", "dxy_beta_cond_60x20", "calmness_20",
       "gain_loss_20", "intraday_drift_20", "usdjpy_beta_cond_120x60",
       "downside_dev_60", "days_since_high_60"]
lib = {}
for e in EFF:
    p = Path("factors") / f"{e}.signal.npy"
    if p.exists():
        a = np.load(p)
        lib[e] = pd.DataFrame(a, index=idx, columns=close.columns)
for n in names:
    row = {e: round(panel_rank_corr(cands[n], lib[e]), 4) for e in lib}
    maxa = max(abs(v) for v in row.values())
    print(f"  {n:20s} max|rho|={maxa:.4f} | " + " ".join(f"{k}={v:+.3f}" for k, v in row.items()))
