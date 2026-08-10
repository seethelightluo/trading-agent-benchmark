"""miner_3 cycle-4 broad screen (2026-07-16).

New candidate families NOT screened before:
  A. Return-distribution shape: skew, kurtosis, downside-semideviation ratio,
     variance ratio (mean-reversion persistence), max drawdown
  B. Range position / candle geometry: stochastic position, candle shadows
  C. Conditional reversals (built on cycle-3 winners rev_intraday_1d, volz_20)
  D. Cross-asset lead-lag / rate sensitivity / cross-sectional dispersion
  E. Liquidity: Amihud illiquidity

Admission gate: |IC1| >= 0.0070 and |ICIR1| >= 0.0840 (1d forward, rank IC,
2021-01-04..2026-07-15, min 8 names/date). For every passing candidate also
report max_abs_library_correlation vs the 4 artifact-bearing library factors.
"""
import time, sys, os, json
import numpy as np
import pandas as pd
import base64, gzip, io
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner1_common import SYMBOLS, load_close

t0 = time.time()
closes = load_close()
idx = None
for s, df in closes.items():
    idx = df.index if idx is None else idx.intersection(df.index)
idx = idx[(idx >= pd.Timestamp("2020-01-01"))]
idx = idx[(idx <= pd.Timestamp("2026-07-15"))]

CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in SYMBOLS})
OP = pd.DataFrame({s: closes[s]["open"].reindex(idx).astype(float) for s in SYMBOLS})
HP = pd.DataFrame({s: closes[s]["high"].reindex(idx).astype(float) for s in SYMBOLS})
LP = pd.DataFrame({s: closes[s]["low"].reindex(idx).astype(float) for s in SYMBOLS})
VO = pd.DataFrame({s: closes[s]["volume"].reindex(idx).astype(float) for s in SYMBOLS})

RET = CP.pct_change()
LRET = np.log(CP / CP.shift(1))
vol5 = LRET.rolling(5).std() * np.sqrt(252)
vol20 = LRET.rolling(20).std() * np.sqrt(252)
vol60 = LRET.rolling(60).std() * np.sqrt(252)
mom20 = CP / CP.shift(20) - 1.0
GATE_IC, GATE_ICIR = 0.0070, 0.0840
EVAL_START = pd.Timestamp("2021-01-04")
HORIZONS = (1, 2, 3, 5, 10, 20)
fwd = {h: RET.shift(-h) for h in HORIZONS}
fwd_ranks = {h: fwd[h].rank(axis=1) for h in HORIZONS}


def row_spearman(F, R):
    X = F.values.astype(float)
    Y = R.values.astype(float)
    valid = (~np.isnan(X)) & (~np.isnan(Y))
    n = valid.sum(axis=1)
    X = np.where(valid, X, np.nan)
    Y = np.where(valid, Y, np.nan)
    with np.errstate(all="ignore"):
        xm = np.nanmean(X, axis=1, keepdims=True)
        ym = np.nanmean(Y, axis=1, keepdims=True)
        xc = np.where(valid, X - xm, 0.0)
        yc = np.where(valid, Y - ym, 0.0)
        num = (xc * yc).sum(axis=1)
        dx = np.sqrt((xc * xc).sum(axis=1))
        dy = np.sqrt((yc * yc).sum(axis=1))
        corr = num / (dx * dy)
    corr = np.where((n >= 8) & np.isfinite(corr), corr, np.nan)
    return pd.Series(corr, index=F.index)


def evaluate(name, fac):
    out = {}
    fr = fac.rank(axis=1)
    for h in HORIZONS:
        s = row_spearman(fr, fwd_ranks[h])
        s = s[(s.index >= EVAL_START)].dropna()
        if len(s) < 120:
            out[h] = None
            continue
        m = float(s.mean())
        sd = float(s.std(ddof=1))
        out[h] = dict(ic=m, icir=m / sd if sd > 1e-12 else 0.0,
                      hit=float((s > 0).mean()), n=int(len(s)))
    sub = fac.loc[fac.index >= EVAL_START]
    cov = float(sub.notna().mean().mean()) if len(sub) else 0.0
    rk = fac.rank(axis=1, pct=True)
    turn = float((rk - rk.shift(10)).abs().mean().mean()) if len(rk) else np.nan
    return dict(horizons=out, coverage=cov, turnover_10d=turn)


# ---------------- library correlation ----------------
def load_artifact(j):
    a = j.get('signal_artifact')
    if a is None:
        return None
    if isinstance(a, str):
        if a.endswith('.npy'):
            p = a if os.path.exists(a) else os.path.join('factors', a)
            return np.load(p, allow_pickle=True)
        return None
    if isinstance(a, dict):
        data = a.get('data') or a.get('matrix') or a.get('encoded')
        if data is None:
            return None
        raw = base64.b64decode(data)
        if raw[:2] == b'\x1f\x8b':
            raw = gzip.decompress(raw)
        return np.load(io.BytesIO(raw))
    return None


LIB = ['factors/miner2_20260716_mom_10d_skip5.json',
       'factors/miner2_20260716_nclv_1d.json',
       'factors/miner3_20260716_rev_intraday_1d.json',
       'factors/miner3_20260716_volz_20.json']
EVAL_IDX = idx[idx >= EVAL_START]
lib_sigs = {}
for f in LIB:
    j = json.load(open(f))
    arr = load_artifact(j)
    if arr is None:
        print(f"  [lib] {f}: no artifact, skip")
        continue
    a = j.get('signal_artifact')
    if isinstance(a, dict) and a.get('n_dates') == len(EVAL_IDX):
        # eval-window aligned (trading days from EVAL_START..end)
        lib_sigs[j['factor_id']] = pd.DataFrame(arr, index=EVAL_IDX, columns=SYMBOLS)
    elif isinstance(a, str) and a.endswith('.npy') and arr.shape[0] >= 2388:
        # calendar-aligned from 2020-01-01
        cal0 = pd.Timestamp("2020-01-01")
        rows = (EVAL_IDX - cal0).days.to_numpy()
        sub = arr[rows]
        lib_sigs[j['factor_id']] = pd.DataFrame(sub, index=EVAL_IDX, columns=SYMBOLS)
    else:
        print(f"  [lib] {f}: unaligned ({arr.shape}), skip")
        continue
    print(f"  [lib] loaded {j['factor_id']} -> {lib_sigs[j['factor_id']].shape}")


def max_lib_corr(fac):
    """max |rank-correlation| (cross-time pooled) vs library signals on eval window."""
    ev = fac.loc[fac.index >= EVAL_START]
    best = 0.0
    for lid, lsig in lib_sigs.items():
        le = lsig.loc[ev.index]
        fv = ev.values.ravel()
        lv = le.values.ravel()
        m = (~np.isnan(fv)) & (~np.isnan(lv))
        if m.sum() < 500:
            continue
        rho = spearmanr(fv[m], lv[m]).statistic
        best = max(best, abs(rho))
    return float(best)


F = {}

# ---------- A. Return-distribution shape ----------
F["skew_20"] = LRET.rolling(20).skew()
F["kurt_20"] = LRET.rolling(20).kurt()
def semidev_ratio(x, win=20):
    neg = x.clip(upper=0.0)
    return neg.rolling(win).std() / (x.rolling(win).std() + 1e-12)
F["semidev_ratio_20"] = semidev_ratio(LRET, 20)
F["var_ratio_5_20"] = LRET.rolling(5).sum().rolling(20).var() / \
                      (LRET.rolling(5).var() * 20 + 1e-12)  # >1 trending, <1 mean-rev
F["var_ratio_10_60"] = LRET.rolling(10).sum().rolling(60).var() / \
                       (LRET.rolling(10).var() * 60 + 1e-12)
def max_dd(x, win=60):
    return (x / x.rolling(win, min_periods=win).max() - 1.0).rolling(win).min()
F["max_dd_60"] = max_dd(CP, 60)

# ---------- B. Range position / candle geometry ----------
F["stoch_10"] = (CP - LP.rolling(10).min()) / (HP.rolling(10).max() - LP.rolling(10).min() + 1e-12) - 0.5
F["stoch_20"] = (CP - LP.rolling(20).min()) / (HP.rolling(20).max() - LP.rolling(20).min() + 1e-12) - 0.5
rng = HP - LP
hi_oc = np.maximum(OP, CP)
lo_oc = np.minimum(OP, CP)
F["upper_shadow_5"] = ((HP - hi_oc) / (rng + 1e-12)).rolling(5).mean()
F["lower_shadow_5"] = ((lo_oc - LP) / (rng + 1e-12)).rolling(5).mean()

# ---------- C. Conditional reversals (cycle-3 winners x state) ----------
rev_intra = 1.0 - CP / OP                      # cycle-3 winner, positive->up next day
volz = (VO - VO.rolling(20).mean()) / (VO.rolling(20).std() + 1e-9)
vol20_pct = vol20.rolling(120).rank(pct=True)
disp = vol20.sub(vol20.mean(axis=1), axis=0).div(vol20.mean(axis=1) + 1e-9, axis=0)  # idiosyncratic vol
F["rev_intra_x_volrank"] = rev_intra * vol20_pct
F["rev_intra_x_disp"] = rev_intra * disp
gap1 = OP / CP.shift(1) - 1.0
F["rev_intra_x_gap_agree"] = rev_intra * np.sign(gap1)          # reversal stronger after gap?
F["rev_intra_x_volz"] = rev_intra * volz
F["volz_x_mom20"] = volz * mom20
F["volz_x_rev_intra"] = volz * rev_intra
F["volz_x_skew20"] = volz * LRET.rolling(20).skew()

# ---------- D. Cross-asset / rate / dispersion ----------
spx = LRET["SPX"]
def roll_beta(y, x, win):
    out = {}
    for s in y.columns:
        df = pd.concat([y[s], x], axis=1).dropna()
        cov = df.iloc[:, 0].rolling(win).cov(df.iloc[:, 1])
        var = df.iloc[:, 1].rolling(win).var()
        out[s] = (cov / (var + 1e-12)).reindex(idx)
    return pd.DataFrame(out)
F["spx_lead_beta_5"] = roll_beta(LRET, spx.shift(1), 5)         # follow SPX next-day move
us10y = LRET["US10Y"] if "US10Y" in LRET else None
if us10y is not None:
    F["us10y_beta_60"] = roll_beta(LRET, us10y, 60)
    F["us10y_beta_60_x_chg20"] = F["us10y_beta_60"].mul(us10y.rolling(20).sum().to_numpy().reshape(-1, 1), axis=0)
F["cross_disp_rank_20"] = vol20.sub(vol20.mean(axis=1), axis=0).div(vol20.mean(axis=1) + 1e-9, axis=0)

# ---------- E. Liquidity ----------
F["amihud_20_neg"] = -(LRET.abs() / (VO + 1e-9)).rolling(20).mean()

results = {name: evaluate(name, fac) for name, fac in F.items()}

print(f"{'factor':26s} {'IC1':>8s} {'ICIR1':>8s} {'hit1':>6s} {'n1':>5s} {'IC5':>8s} {'IC10':>8s} {'IC20':>8s} {'cov':>6s} {'turn10':>7s} {'libCorr':>7s}  gate")
rows = []
for name, r in results.items():
    h = r["horizons"]
    g = h.get(1)
    if g is None:
        continue
    h5, h10, h20 = h.get(5), h.get(10), h.get(20)
    passed = abs(g["ic"]) >= GATE_IC and abs(g["icir"]) >= GATE_ICIR
    lc = max_lib_corr(F[name]) if passed else float("nan")
    rows.append((name, g["ic"], g["icir"], g["hit"], g["n"],
                 h5["ic"] if h5 else np.nan, h10["ic"] if h10 else np.nan,
                 h20["ic"] if h20 else np.nan, r["coverage"], r["turnover_10d"], lc, passed))
rows.sort(key=lambda x: -abs(x[1]))
for r in rows:
    mark = "PASS" if r[-1] else "   "
    print(f"{r[0]:26s} {r[1]:8.4f} {r[2]:8.4f} {r[3]:6.3f} {r[4]:5d} {r[5]:8.4f} {r[6]:8.4f} {r[7]:8.4f} {r[8]:6.3f} {r[9]:7.3f} {r[10]:7.3f}  {mark}")

print(f"\n{len(idx)} dates x {len(SYMBOLS)} symbols | {len(F)} candidates | "
      f"{sum(r[-1] for r in rows)} passed gate | {time.time()-t0:.1f}s")
with open("scripts/miner3_screen_cycle4_results.json", "w") as fh:
    json.dump({n: {"horizons": {str(k): v for k, v in r["horizons"].items()},
                   "coverage": r["coverage"], "turnover_10d": r["turnover_10d"]}
               for n, r in results.items()}, fh, indent=1, default=str)
print("saved scripts/miner3_screen_cycle4_results.json")
