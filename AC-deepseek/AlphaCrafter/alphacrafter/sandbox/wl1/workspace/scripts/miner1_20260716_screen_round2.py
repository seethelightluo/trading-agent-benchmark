"""miner_1 round 2: refine trend-efficiency (er) family + macro-beta factors with
better rolling coverage (min_periods), weekday-aligned panel.

Gates: |IC1|>=0.0070, |ICIR1|>=0.0840 on 2021-01-01..2026-07-15.
"""
import numpy as np
import pandas as pd

panel = pd.read_pickle('scripts/panel_cache.pkl')
close = panel['close']
opn = panel['open']
high = panel['high']
low = panel['low']
ret = panel['ret']
macro = panel['macro']

idx = close.index[close.index.dayofweek < 5]
idx = idx[(idx >= pd.Timestamp('2020-01-01')) & (idx <= pd.Timestamp('2026-07-15'))]
VAL = idx[idx >= pd.Timestamp('2021-01-01')]
SYMS = list(close.columns)

C = close.reindex(idx).astype(float)
O = opn.reindex(idx).astype(float)
H = high.reindex(idx).astype(float)
L = low.reindex(idx).astype(float)
R = ret.reindex(idx).astype(float)
LR = np.log(C / C.shift(1))
VOL20 = R.rolling(20, min_periods=10).std()
VOL60 = R.rolling(60, min_periods=30).std()

fwd = {h: C.shift(-h) / C - 1.0 for h in (1, 2, 3, 5, 10, 20, 30)}

M = {}
for m in macro.columns:
    s = macro[m].reindex(idx).ffill().astype(float)
    M[m] = s
DVIX = np.log(M['VIX'] / M['VIX'].shift(1))
DDXY = np.log(M['DXY'] / M['DXY'].shift(1))
DJPY = np.log(M['USDJPY'] / M['USDJPY'].shift(1))
DBTC = np.log(C['BTC'] / C['BTC'].shift(1))


def rolling_beta(x, f, win, mp):
    cov = x.rolling(win, min_periods=mp).cov(f)
    var = f.rolling(win, min_periods=mp).var()
    return cov.div(var.replace(0, np.nan), axis=0)


def eff_ratio(nd, mp=10, skip=0):
    """Efficiency ratio over nd days with optional skip: |net move| / sum(|moves|)."""
    if skip > 0:
        net = (C.shift(skip) / C.shift(skip + nd) - 1.0).abs()
        gross = LR.abs().rolling(skip + nd, min_periods=mp).sum() - LR.abs().rolling(skip, min_periods=mp).sum()
    else:
        net = (C / C.shift(nd) - 1.0).abs()
        gross = LR.abs().rolling(nd, min_periods=mp).sum()
    return net / (gross + 1e-12)


cands = {}
for nd in (10, 20, 30):
    cands[f'er{nd}'] = eff_ratio(nd)
cands['er20_skip5'] = eff_ratio(20, skip=5)
cands['er10_skip5'] = eff_ratio(10, skip=5)
cands['er20_voladj'] = eff_ratio(20) / (VOL20 + 1e-12)

# macro betas with improved coverage
for win, mp in ((60, 30), (40, 25)):
    cands[f'usdjpy_beta{win}'] = rolling_beta(LR, DJPY, win, mp)
    cands[f'dxy_beta{win}'] = rolling_beta(LR, DDXY, win, mp)
    cands[f'vix_beta{win}'] = rolling_beta(LR, DVIX, win, mp)
    cands[f'btc_beta{win}'] = rolling_beta(LR, DBTC, win, mp)

# world ex-self beta (fixed broadcast)
n_valid = R.notna().sum(axis=1)
world_ex = R.rsub(R.sum(axis=1), axis=0).div((n_valid - 1).replace(0, np.nan), axis=0)
cands['world_beta60'] = rolling_beta(LR, world_ex, 60, 30)
cands['world_beta40'] = rolling_beta(LR, world_ex, 40, 25)

# updown ratio with coverage fix
cands['updown20'] = R.clip(lower=0).rolling(20, min_periods=10).sum() / \
    ((-R).clip(lower=0).rolling(20, min_periods=10).sum() + 1e-12)

N_CELLS = len(VAL) * len(SYMS)
GATE_IC, GATE_ICIR = 0.0070, 0.0840

lib = {}
lib['mom10_skip5'] = np.log(C / C.shift(10)) - np.log(C / C.shift(5))
lib['mom120_skip5'] = C.shift(5) / C.shift(125) - 1.0
lib['rev1'] = -R
lib['nclv1'] = -(C - L) / (H - L).replace(0, np.nan)
lib['intraday_rev'] = 1.0 - C / O
lib['vol_of_vol'] = R.rolling(20, min_periods=10).std().rolling(60, min_periods=30).std()
lib['vix_beta_cond'] = -rolling_beta(LR, DVIX, 60, 30).mul(M['VIX'] / M['VIX'].shift(20) - 1.0, axis=0)
volz_vol = (VOL20 - VOL20.rolling(120, min_periods=60).mean()) / (VOL20.rolling(120, min_periods=60).std() + 1e-12)
lib['volz20'] = volz_vol


def panel_corr(a, b):
    A = a.values.astype(float)
    B = b.values.astype(float)
    m = np.isfinite(A) & np.isfinite(B)
    if int(m.sum()) < 50:
        return np.nan
    x, y = A[m], B[m]
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


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
        Fv = F.values.astype(float)
        Rv = Rf.values.astype(float)
        mask = np.isfinite(Fv) & np.isfinite(Rv)
        n = mask.sum(axis=1)
        ok = n >= 8
        if not ok.any():
            ics[h] = {'ic': np.nan, 'icir': np.nan, 'n': 0}
            continue
        Fm = np.where(mask, Fv, 0.0)
        Rm = np.where(mask, Rv, 0.0)
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
                  'n': int(len(ic))}
    ic1 = ics[1]
    passed = (abs(ic1['ic']) >= GATE_IC) and (abs(ic1['icir']) >= GATE_ICIR)
    corrs = [panel_corr(p, lv) for lv in lib.values()]
    corrs = [c for c in corrs if c is not None and np.isfinite(c)]
    maxc = max(abs(c) for c in corrs) if corrs else np.nan
    dec = ' '.join(f"h{h}:{ics[h]['ic']:+.3f}" for h in (2, 3, 5, 10, 20))
    print(f"{name:16s} cov={cov:.2f} to={to:.2f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} "
          f"n1={ic1['n']} | libCorr={maxc:.2f} | {dec} | {'PASS' if passed else 'fail'}")
    return {'name': name, 'panel': p, 'cov': cov, 'to': to, 'ics': ics,
            'passed': passed, 'max_lib_corr': maxc}


print(f"{'='*110}")
res = {}
for nm, p in cands.items():
    try:
        res[nm] = evaluate(nm, p)
    except Exception as e:
        print(f"{nm}: ERROR {e}")

passers = {k: v for k, v in res.items() if v['passed']}
print(f"\nTotal candidates: {len(cands)}, PASS: {len(passers)} -> {list(passers.keys())}")

# by-year IC1 for passers
for nm in passers:
    p = passers[nm]['panel']
    yr = {}
    for y in range(2021, 2027):
        m = (VAL >= pd.Timestamp(f'{y}-01-01')) & (VAL <= pd.Timestamp(f'{y}-12-31'))
        if m.sum() < 30:
            continue
        F = p.reindex(VAL[m]).rank(axis=1)
        Rf = fwd[1].reindex(VAL[m]).rank(axis=1)
        Fv, Rv = F.values.astype(float), Rf.values.astype(float)
        mask = np.isfinite(Fv) & np.isfinite(Rv)
        n = mask.sum(axis=1)
        ok = n >= 8
        if not ok.any():
            continue
        Fm = np.where(mask, Fv, 0.0)
        Rm = np.where(mask, Rv, 0.0)
        sx, sy = Fm.sum(1), Rm.sum(1)
        sxx, syy, sxy = (Fm * Fm).sum(1), (Rm * Rm).sum(1), (Fm * Rm).sum(1)
        with np.errstate(all='ignore'):
            num = n * sxy - sx * sy
            den = np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
            ic = num / den
        ic = ic[ok]
        ic = ic[np.isfinite(ic)]
        yr[y] = {'ic': round(float(ic.mean()), 4) if len(ic) else None,
                 'n': int(len(ic))}
    print(f"{nm:16s} by_year={yr}")
