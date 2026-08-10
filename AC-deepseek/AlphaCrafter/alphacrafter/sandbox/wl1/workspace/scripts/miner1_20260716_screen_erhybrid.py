"""miner_1 screen: ER-gated momentum/reversal hybrids (trend-quality interaction).

Hypothesis: smooth directional trends (high Kaufman ER) persist cross-sectionally,
so momentum should be amplified in high-ER regimes and reversal should dominate in
choppy (low-ER) regimes. Candidates combine the persisted er20 signal with
momentum/reversal variants. Validation on 2021-01-01..2026-07-15, 15-name panel.
Gate: |IC1|>=0.007, |ICIR1|>=0.084, coverage>=0.4, max lib |rho|<0.5.
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner1_common import (SYMBOLS, MACRO, CUT, START, load_close,
                           ic_analysis, decay_analysis, coverage, turnover)

VAL_START = pd.Timestamp("2021-01-01")
GATE_IC, GATE_ICIR = 0.007, 0.084
N_CELLS = 15

closes = load_close()
macros = load_close(MACRO, dir_="../persistent/index_data")

# ---- factor builders (all use data through t only) ----
def build_panel(fn):
    cols = {}
    for s in SYMBOLS:
        df = closes[s]
        try:
            fv = fn(df)
            if fv is not None and len(fv):
                cols[s] = fv
        except Exception as e:
            print(f"  [warn] {s}: {e}")
    return pd.DataFrame(cols)

def logret(df, n=1):
    return np.log(df["close"] / df["close"].shift(n))

def er_series(df, win=20):
    net = (df["close"] / df["close"].shift(win) - 1.0).abs()
    path = np.log(df["close"] / df["close"].shift(1)).abs().rolling(win).sum()
    return net / (path + 1e-12)

def mom_skip(df, win=20, skip=5):
    return df["close"] / df["close"].shift(win + skip) - 1.0

def rev_series(df, win=5):
    return df["close"] / df["close"].shift(win) - 1.0

def cs_median(panel):
    return panel.median(axis=1)

# ---- candidate panels ----
cands = {}

# 1) momentum20_skip5 x binary ER gate (ER > 0.25)
p = {}
for s in SYMBOLS:
    df = closes[s]
    er = er_series(df, 20)
    mom = mom_skip(df, 20, 5)
    p[s] = mom * (er > 0.25).astype(float)
cands["mom20x_er_gate25"] = pd.DataFrame(p)

# 2) momentum20_skip5 x soft ER offset (ER - 0.2), clip at 0
p = {}
for s in SYMBOLS:
    df = closes[s]
    er = er_series(df, 20)
    mom = mom_skip(df, 20, 5)
    w = (er - 0.2).clip(lower=0)
    p[s] = mom * w
cands["mom20x_er_soft"] = pd.DataFrame(p)

# 3) momentum20 x ER centering by cross-sectional expanding median
er_panel = build_panel(lambda df: er_series(df, 20))
mom_panel = build_panel(lambda df: mom_skip(df, 20, 5))
er_med = er_panel.median(axis=1).expanding(min_periods=60).median()
cands["mom20x_er_csmed"] = mom_panel.mul(er_panel.sub(er_med, axis=0), axis=0)

# 4) reversal 5d gated by LOW ER (choppy regime -> reversal): factor = -ret5 * (ER < 0.15)
p = {}
for s in SYMBOLS:
    df = closes[s]
    er = er_series(df, 20)
    rev = rev_series(df, 5)
    p[s] = (-rev) * (er < 0.15).astype(float)
cands["rev5x_er_low"] = pd.DataFrame(p)

# 5) reversal 5d x (1 - ER) soft weight (anti-trend weight)
p = {}
for s in SYMBOLS:
    df = closes[s]
    er = er_series(df, 20)
    rev = rev_series(df, 5)
    p[s] = (-rev) * (1.0 - er).clip(lower=0)
cands["rev5x_er_soft"] = pd.DataFrame(p)

# 6) cross-sectional rank product: rank(mom20_skip5) * rank(er20)
cands["rankprod_mom_er"] = mom_panel.rank(axis=1).mul(er_panel.rank(axis=1))

# 7) er20 x mom120_skip5 (long-horizon momentum x trend quality)
p = {}
for s in SYMBOLS:
    df = closes[s]
    er = er_series(df, 20)
    mom120 = mom_skip(df, 120, 5)
    p[s] = mom120 * er
cands["mom120x_er"] = pd.DataFrame(p)

# 8) signed trend-quality: ER x sign(momentum20_skip5) -> amplitude = |mom| in smooth trends
p = {}
for s in SYMBOLS:
    df = closes[s]
    er = er_series(df, 20)
    mom = mom_skip(df, 20, 5)
    p[s] = np.sign(mom) * er * mom.abs()
cands["er_signed_amp"] = pd.DataFrame(p)

# ---- library artifacts for collinearity check ----
def load_lib_artifacts():
    arts = {}
    for f in sorted(os.listdir("factors")):
        if f.endswith(".npy"):
            arts[f.replace(".npy", "")] = np.load(os.path.join("factors", f))
    return arts

lib_arts = load_lib_artifacts()
print("library artifacts:", list(lib_arts.keys()))

def library_corr(panel):
    """max |spearman rho| of panel vs each library artifact (aligned by position)."""
    maxc = np.nan
    corrs = []
    base = panel.values
    for name, arr in lib_arts.items():
        n = min(base.shape[0], arr.shape[0])
        x = base[:n].astype(float)
        y = arr[:n].astype(float)
        # row-wise spearman via rank
        from scipy.stats import spearmanr
        rhos = []
        for i in range(n):
            a = x[i]
            b = y[i]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 4:
                continue
            r = spearmanr(a[m], b[m])[0]
            if np.isfinite(r):
                rhos.append(r)
        r = float(np.mean(rhos)) if rhos else np.nan
        corrs.append((name, r))
        if np.isfinite(r):
            maxc = r if not np.isfinite(maxc) else max(maxc, abs(r))
    return maxc, corrs

# ---- evaluation ----
VAL = {}
results = {}
for s in SYMBOLS:
    df = closes[s]
    d = df[df.index >= VAL_START]
    VAL[s] = df

def evaluate(name, panel):
    panel = panel[panel.index >= VAL_START]
    cov = coverage(panel, closes)
    to = turnover(panel)
    ics = {h: ic_analysis(panel, closes, fwd_days=h) for h in (1, 2, 3, 5, 10, 20)}
    ic1 = ics[1]["ic"]
    icir1 = ics[1]["icir"]
    maxc, corrs = library_corr(panel)
    by_year = {}
    for yr in range(2021, 2027):
        sub = panel[(panel.index >= pd.Timestamp(f"{yr}-01-01")) & (panel.index <= pd.Timestamp(f"{yr}-12-31"))]
        r = ic_analysis(sub, closes, fwd_days=1)
        by_year[yr] = {"ic": r["ic"], "icir": r["icir"], "n": r["n_dates"]}
    passed = (abs(ic1) >= GATE_IC and abs(icir1) >= GATE_ICIR and cov >= 0.4
              and (not np.isfinite(maxc) or maxc < 0.5))
    dec = " ".join(f"h{h}:{ics[h]['ic']:+.3f}" for h in (2, 3, 5, 10, 20))
    print(f"{name:18s} cov={cov:.2f} to={to:.2f} | IC1={ic1:+.4f} ICIR1={icir1:+.3f} "
          f"hit={ics[1]['hit']:.2f} n1={ics[1]['n_dates']} | libCorr={maxc if np.isfinite(maxc) else float('nan'):.3f} | {dec} | {'PASS' if passed else 'fail'}")
    if corrs:
        worst = sorted(corrs, key=lambda t: -abs(t[1]))[:3]
        print(f"           top lib corrs: {[(k, round(rr, 3)) for k, rr in worst]}")
    return {"cov": cov, "turnover": to, "ics": ics, "passed": passed,
            "max_lib_corr": maxc, "by_year": by_year, "lib_corrs": corrs}

print(f"\nCandidates: {len(cands)}, gate IC>={GATE_IC} ICIR>={GATE_ICIR}, cov>=0.4, window {VAL_START.date()}..{CUT.date()}")
print("=" * 130)
for nm, p in cands.items():
    results[nm] = evaluate(nm, p)

passers = {k: v for k, v in results.items() if v["passed"] and v["cov"] >= 0.4}
print(f"\nTotal candidates: {len(cands)}, PASS: {len(passers)} -> {list(passers.keys())}")

with open("scripts/miner1_erhybrid_results.pkl", "wb") as fh:
    pickle.dump({k: {kk: vv for kk, vv in v.items() if kk != "panel"} for k, v in results.items()}, fh)
print("results saved to scripts/miner1_erhybrid_results.pkl")
