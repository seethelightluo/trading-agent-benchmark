"""miner_1: deep validation of the trend-efficiency (Kaufman ER) factor family.

Candidate idea: er20 = |20d net return| / (sum of |daily log returns| over 20d).
High ER => smooth directional trend (price moved far with little noise);
low ER => choppy/market. Hypothesis: cross-sectionally, smooth trends
persist (continuation), so higher ER predicts higher forward returns.

Admission gate (15-name cross-asset universe, daily rank IC at h=1):
    |IC1| >= 0.0070 and |ICIR1| >= 0.0840
Validation window: 2021-01-01..2026-07-15 (2020 warm-up).
"""
import numpy as np
import pandas as pd
import json, base64, zlib, glob, os

panel = pd.read_pickle('scripts/panel_cache.pkl')
close = panel['close']
opn = panel['open']
high = panel['high']
low = panel['low']

idx_all = close.index[close.index.dayofweek < 5]
idx = idx_all[(idx_all >= pd.Timestamp('2020-01-01')) & (idx_all <= pd.Timestamp('2026-07-15'))]
VAL = idx[idx >= pd.Timestamp('2021-01-01')]
SYMS = list(close.columns)

C = close.reindex(idx).astype(float)
O = opn.reindex(idx).astype(float)
H = high.reindex(idx).astype(float)
L = low.reindex(idx).astype(float)
LR = np.log(C / C.shift(1))

fwd = {}
for h in (1, 2, 3, 5, 10, 20, 30):
    fwd[h] = C.shift(-h) / C - 1.0

N_CELLS = len(VAL) * len(SYMS)
GATE_IC, GATE_ICIR = 0.0070, 0.0840

# ---------------- candidate family ----------------
cands = {}
def er(win, mp):
    num = (C / C.shift(win) - 1.0).abs()
    den = LR.abs().rolling(win, min_periods=mp).sum()
    return num / (den + 1e-12)

cands['er20'] = er(20, 20)
cands['er20_mp10'] = er(20, 10)
cands['er10'] = er(10, 10)
cands['er40'] = er(40, 40)
cands['er60'] = er(60, 40)
cands['er20_sk5'] = (C.shift(5) / C.shift(25) - 1.0).abs() / LR.abs().shift(5).rolling(20, min_periods=10).sum().add(1e-12)

# ---------------- library artifact loader ----------------
def decode_artifact(sig):
    """sig: dict with format 'base64:zlib:csv' or str filename -> DataFrame"""
    if isinstance(sig, str):
        a = np.load(sig, allow_pickle=True)
        return pd.DataFrame(a, index=close.index, columns=SYMS)
    if isinstance(sig, dict) and sig.get('format') == 'base64:zlib:csv':
        raw = base64.b64decode(sig['data'])
        txt = zlib.decompress(raw).decode()
        df = pd.read_csv(pd.io.common.StringIO(txt), index_col=0)
        df.index = pd.to_datetime(df.index)
        return df
    return None

def load_library_signals():
    out = {}
    # .npy artifacts
    for f in glob.glob('factors/*.npy'):
        nm = os.path.basename(f).replace('.npy', '')
        try:
            a = np.load(f)
            df = pd.DataFrame(a, index=close.index, columns=SYMS)
            out[nm] = df
        except Exception as e:
            print('  npy load fail', f, e)
    # embedded base64 artifacts in top-level JSONs
    for f in glob.glob('factors/*.json'):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        nm = d.get('factor_id', os.path.basename(f))
        if nm in out:
            continue
        cand = None
        v = d.get('validation', {})
        if isinstance(v.get('signal_artifact'), dict):
            cand = decode_artifact(v['signal_artifact'])
        if cand is not None and cand.shape[0] > 100:
            out[nm] = cand
    return out

LIB = load_library_signals()
print(f'library artifacts loaded: {len(LIB)} -> {list(LIB.keys())}')

def spearman_rho(a, b):
    A = a.values.astype(float).ravel()
    B = b.values.astype(float).ravel()
    m = np.isfinite(A) & np.isfinite(B)
    if int(m.sum()) < 50:
        return np.nan
    x, y = A[m], B[m]
    if np.unique(x).size < 3 or np.unique(y).size < 3:
        return np.nan
    return float(pd.Series(x).corr(pd.Series(y), method='spearman'))

# ---------------- evaluation ----------------
def evaluate(name, p):
    p = p.reindex(idx)
    cov = float(p.reindex(VAL).notna().sum().sum()) / N_CELLS
    ranks = p.rank(axis=1)
    to = []
    for i in range(10, len(ranks)):
        prev = ranks.iloc[i - 10].dropna()
        cur = ranks.iloc[i].dropna()
        cmn = prev.index.intersection(cur.index)
        if len(cmn) >= 2:
            to.append((cur[cmn] - prev[cmn]).abs().mean() / (len(cmn) - 1))
    to = float(np.mean(to)) if to else np.nan

    ics = {}
    for h in (1, 2, 3, 5, 10, 20, 30):
        F = p.reindex(VAL).rank(axis=1)
        Rf = fwd[h].reindex(VAL).rank(axis=1)
        Fv = F.values.astype(float); Rv = Rf.values.astype(float)
        mask = np.isfinite(Fv) & np.isfinite(Rv)
        n = mask.sum(axis=1)
        ok = n >= 8
        if not ok.any():
            ics[h] = {'ic': np.nan, 'icir': np.nan, 'hit': np.nan, 'n': 0}
            continue
        Fm = np.where(mask, Fv, 0.0); Rm = np.where(mask, Rv, 0.0)
        sx, sy = Fm.sum(1), Rm.sum(1)
        sxx, syy, sxy = (Fm * Fm).sum(1), (Rm * Rm).sum(1), (Fm * Rm).sum(1)
        with np.errstate(all='ignore'):
            num = n * sxy - sx * sy
            den = np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
            ic = num / den
        ic = ic[ok]
        ic = ic[np.isfinite(ic)]
        ics[h] = {'ic': float(ic.mean()) if len(ic) else np.nan,
                  'icir': float(ic.mean() / ic.std()) if len(ic) > 1 and ic.std() > 0 else np.nan,
                  'hit': float((ic > 0).mean()) if len(ic) else np.nan,
                  'n': int(len(ic))}

    ic1, icir1 = ics[1]['ic'], ics[1]['icir']
    passed = (abs(ic1) >= GATE_IC) and (abs(icir1) >= GATE_ICIR)

    # library correlation on VAL-aligned signals
    corrs = []
    for nm, lv in LIB.items():
        lv_al = lv.reindex(VAL)
        r = spearman_rho(p.reindex(VAL), lv_al)
        if r is not None and np.isfinite(r):
            corrs.append((nm, r))
    maxc = max(abs(r) for _, r in corrs) if corrs else np.nan

    # by-year IC1
    by_year = {}
    Fv = p.reindex(VAL).rank(axis=1).values.astype(float)
    Rv = fwd[1].reindex(VAL).rank(axis=1).values.astype(float)
    mask = np.isfinite(Fv) & np.isfinite(Rv)
    n = mask.sum(axis=1)
    Fm = np.where(mask, Fv, 0.0); Rm = np.where(mask, Rv, 0.0)
    sx, sy = Fm.sum(1), Rm.sum(1)
    sxx, syy, sxy = (Fm * Fm).sum(1), (Rm * Rm).sum(1), (Fm * Rm).sum(1)
    with np.errstate(all='ignore'):
        num = n * sxy - sx * sy
        den = np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
        ic = num / den
    for yr in range(2021, 2027):
        ym = VAL.year == yr
        ics_yr = ic[ym]
        ics_yr = ics_yr[np.isfinite(ics_yr)]
        if len(ics_yr):
            by_year[str(yr)] = {'ic': float(ics_yr.mean()),
                                'icir': float(ics_yr.mean() / ics_yr.std()) if ics_yr.std() > 0 else np.nan,
                                'n': int(len(ics_yr))}

    dec = ' '.join(f"h{h}:{ics[h]['ic']:+.3f}" for h in (2, 3, 5, 10, 20))
    print(f"{name:12s} cov={cov:.2f} to={to:.2f} | IC1={ic1:+.4f} ICIR1={icir1:+.3f} hit={ics[1]['hit']:.2f} n1={ics[1]['n']} | "
          f"libCorr={maxc:.2f}{'<' if not np.isfinite(maxc) or maxc < 0.5 else ' >=0.5!'} | {dec} | {'PASS' if passed else 'fail'}")
    if corrs:
        worst = sorted(corrs, key=lambda t: -abs(t[1]))[:3]
        print(f"           top lib corrs: {[(k, round(r, 3)) for k, r in worst]}")
    return {'name': name, 'panel': p, 'cov': cov, 'to': to, 'ics': ics,
            'passed': passed, 'max_lib_corr': maxc, 'by_year': by_year,
            'lib_corrs': corrs}

print(f"\nweekday VAL rows: {len(VAL)}, cells: {N_CELLS}, gate: IC>={GATE_IC}, ICIR>={GATE_ICIR}")
print('=' * 130)
res = {}
for nm, p in cands.items():
    res[nm] = evaluate(nm, p)

passers = {k: v for k, v in res.items() if v['passed'] and v['cov'] >= 0.4}
print(f"\nTotal candidates: {len(cands)}, PASS(gate & cov>=0.4): {len(passers)} -> {list(passers.keys())}")

# dump results for persistence step
import pickle
with open('scripts/miner1_er20_results.pkl', 'wb') as fh:
    pickle.dump({k: {kk: vv for kk, vv in v.items() if kk != 'panel'} for k, v in res.items()}, fh)
print('results saved to scripts/miner1_er20_results.pkl')
