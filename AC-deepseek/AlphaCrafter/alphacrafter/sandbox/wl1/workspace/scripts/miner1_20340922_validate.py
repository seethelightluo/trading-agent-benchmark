"""miner_1 2034-09-22: numpy-vectorized deep validation of ddepth_60d & dskew_60d.

Replaces the slow pandas per-date .loc/.rank().corr() loop with numpy rankdata
IC over pre-extracted matrices. Panel: scripts/panel_cache_20340922.pkl (data
through 2034-09-21, the latest completed trading day).
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import base64, zlib, hashlib

panel = pd.read_pickle('scripts/panel_cache_20340922.pkl')
close = panel['close']; opn = panel['open']; high = panel['high']; low = panel['low']
ret = panel['ret']; macro = panel['macro']
LIVE = [c for c in close.columns if c not in ('HSI', 'CN10Y')]
ALL = list(close.columns)
START = pd.Timestamp('2021-01-01')
IDX = close.index
COL_IDX = {c: i for i, c in enumerate(ALL)}
N_DATES = len(IDX)
N_SYM = len(ALL)
print(f"panel: {N_DATES} dates {IDX.min().date()}..{IDX.max().date()}, {N_SYM} symbols; LIVE={len(LIVE)}")

# ---------------- candidate signals ----------------
ddepth = 1.0 - close / close.rolling(60).max()
r2 = ret.clip(upper=0.0) ** 2
rsum = (ret ** 2).rolling(60).sum()
dskew = r2.rolling(60).sum() / rsum
cands = {'ddepth_60d': ddepth, 'dskew_60d': dskew}

# ---------------- library reconstruction (for correlation audit) ----------------
lib = {}
lib['nclv_1d'] = -(close - low) / (high - low).replace(0, np.nan)
lib['rev_2d'] = -(np.log(close) - np.log(close.shift(2)))
lib['rev_5d'] = -(np.log(close) - np.log(close.shift(5)))
lib['vol_of_vol20x60'] = close.pct_change().rolling(20).std().rolling(60).std()
lib['mom_120d_skip5'] = close.shift(5) / close.shift(125) - 1.0
vix = macro['VIX']; vixr = vix.pct_change(); vixm = vix / vix.shift(20) - 1.0
asset_ret = close.pct_change()
beta60 = asset_ret.rolling(60).cov(vixr) / vixr.rolling(60).var()
lib['vix_beta_cond_60x20'] = -beta60 * vixm

def fwd_ret(h):
    return (close.shift(-h) / close - 1.0)[ALL].to_numpy()

def ic_series_np(S, F, min_n=8):
    """S, F: (n_dates, n_cols) numpy arrays. Spearman rank IC per date."""
    ics = np.empty(len(S)); valid = np.zeros(len(S), dtype=bool)
    for i in range(len(S)):
        s = S[i]; f = F[i]
        m = ~(np.isnan(s) | np.isnan(f))
        if m.sum() < min_n:
            continue
        rs = rankdata(s[m]); rf = rankdata(f[m])
        rs = rs - rs.mean(); rf = rf - rf.mean()
        denom = np.sqrt((rs ** 2).sum() * (rf ** 2).sum())
        if denom == 0:
            continue
        ics[i] = (rs * rf).sum() / denom
        valid[i] = True
    return ics[valid]

def evaluate(sig, cols, horizons, start):
    idx0 = np.asarray(sig.index >= start)
    cols_idx = [COL_IDX[c] for c in cols]
    S = sig.loc[idx0, cols].to_numpy()
    out = {}
    for h in horizons:
        F = fwd_ret(h)[idx0.to_numpy(), :][:, cols_idx]
        ics = ic_series_np(S, F)
        n = len(ics)
        if n == 0:
            out[h] = (np.nan, np.nan, np.nan, 0); continue
        ic = float(ics.mean()); sd = float(ics.std(ddof=1))
        out[h] = (ic, ic / sd if sd > 0 else np.nan, float((ics > 0).mean()), n)
    return out

def show(res, label):
    parts = [f"{label}:"]
    for h in (1, 2, 3, 5, 10, 20):
        ic, icir, hit, n = res[h]
        parts.append(f"h{h} IC={ic:.4f} ICIR={icir:.3f} hit={hit:.2f} n={n}")
    print("  " + " | ".join(parts))

def coverage_stats(sig, cols):
    sub = sig.loc[sig.index >= START, cols]
    cov_ad = float(sub.notna().to_numpy().mean())
    n_d8 = int((sub.notna().sum(axis=1) >= 8).sum())
    return cov_ad, n_d8 / len(sub), len(sub)

def turnover_10d(sig, cols):
    sub = sig.loc[sig.index >= START, cols]
    tot, cnt = 0.0, 0
    for c in cols:
        s = sub[c].dropna()
        if len(s) < 20:
            continue
        r = s.rank(pct=True)
        tot += (r - r.shift(10)).abs().mean()
        cnt += 1
    return tot / cnt if cnt else np.nan

def max_lib_corr(sig, cols):
    idxm = sig.index >= START
    a = sig.loc[idxm, cols]
    best = (0.0, None)
    for lname, lsig in lib.items():
        b = lsig.loc[idxm, cols]
        am = a.to_numpy(); bm = b.to_numpy()
        m = ~(np.isnan(am) | np.isnan(bm))
        if m.sum() < 500:
            continue
        av = am[m]; bv = bm[m]
        rho = float(np.corrcoef(av, bv)[0, 1])
        if abs(rho) > abs(best[0]):
            best = (rho, lname)
    return best

HZ = (1, 2, 3, 5, 10, 20)
print("=" * 110)
summary = {}
for cname, sig in cands.items():
    print(f"\n### {cname} (direction: +)")
    show(evaluate(sig, ALL, HZ, START), "FULL-15")
    show(evaluate(sig, LIVE, HZ, START), "LIVE-13")
    cov_ad, cov_d8, nd = coverage_stats(sig, ALL)
    to = turnover_10d(sig, ALL)
    print(f"  coverage: asset_days={cov_ad:.3f} dates_ge8={cov_d8:.3f} total_dates={nd}  turnover_10d={to:.3f}")
    rho, arg = max_lib_corr(sig, ALL)
    print(f"  max_abs_library_correlation: {rho:+.3f} vs {arg}")
    show(evaluate(sig, LIVE, HZ, pd.Timestamp('2033-10-01')), "RECENT(LIVE,>=2033-10-01)")
    for p0, p1 in [(pd.Timestamp('2021-01-01'), pd.Timestamp('2022-12-31')),
                   (pd.Timestamp('2023-01-01'), pd.Timestamp('2024-12-31')),
                   (pd.Timestamp('2025-01-01'), pd.Timestamp('2026-12-31')),
                   (pd.Timestamp('2027-01-01'), pd.Timestamp('2028-12-31')),
                   (pd.Timestamp('2029-01-01'), pd.Timestamp('2030-12-31')),
                   (pd.Timestamp('2031-01-01'), pd.Timestamp('2032-12-31')),
                   (pd.Timestamp('2033-01-01'), pd.Timestamp('2034-09-21'))]:
        res = evaluate(sig, LIVE, HZ, p0)
        summary.setdefault(cname, {})[str(p0.date())] = {str(h): res[h] for h in HZ}
        print(f"  sub {p0.date()}..{p1.date()}: " + " ".join(
            f"h{h} IC={res[h][0]:.4f} ICIR={res[h][1]:.3f} n={res[h][3]}" for h in (5, 10)))
    summary.setdefault(cname, {})['__full__'] = {str(h): evaluate(sig, LIVE, HZ, START)[h] for h in HZ}
    summary.setdefault(cname, {})['__recent__'] = {str(h): evaluate(sig, LIVE, HZ, pd.Timestamp('2033-10-01'))[h] for h in HZ}

print("\n### mutual correlation ddepth_60d vs dskew_60d (flattened, LIVE, since 2021)")
idxm = close.index >= START
a = ddepth.loc[idxm, LIVE].to_numpy(); b = dskew.loc[idxm, LIVE].to_numpy()
m = ~(np.isnan(a) | np.isnan(b))
print(f"  rho = {float(np.corrcoef(a[m], b[m])[0, 1]):+.3f}  (n={m.sum()})")

def artifact(sig):
    sub = sig.loc[:, ALL]
    csv = sub.round(10).to_csv()
    raw = zlib.compress(csv.encode('utf-8'))
    b64 = base64.b64encode(raw).decode('ascii')
    return b64, sub.shape, int(sub.notna().to_numpy().sum()), hashlib.sha256(csv.encode('utf-8')).hexdigest()[:16]

print("\n### artifact sizes:")
art = {}
for cname, sig in cands.items():
    b64, shp, nv, hh = artifact(sig)
    art[cname] = (b64, shp, nv, hh)
    print(f"  {cname}: shape={shp} n_valid={nv} sha={hh} b64_len={len(b64)}")

# persist intermediate summary for the persistence step
import json
sum_out = {cname: d for cname, d in summary.items()}
sum_out['_artifact'] = {k: {'shape': list(v[1]), 'n_valid': v[2], 'sha': v[3], 'b64_len': len(v[0])} for k, v in art.items()}
with open('scripts/miner1_20340922_summary.json', 'w') as f:
    json.dump(sum_out, f, default=str, indent=1)
print("\nsaved scripts/miner1_20340922_summary.json")
