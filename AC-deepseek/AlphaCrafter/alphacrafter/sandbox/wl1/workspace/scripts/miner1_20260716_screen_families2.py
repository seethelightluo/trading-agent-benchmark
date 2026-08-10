"""miner_1: screen NEW factor families (crash-risk, macro-beta, vol-dynamics,
trend-quality, seasonality) on the clean weekday-aligned 15-asset panel.

Admission gate (15-name cross-asset universe, daily rank IC):
    |IC1| >= 0.0070 and |ICIR1| >= 0.0840
Validation window: 2021-01-01 .. 2026-07-15 (2020 used as warm-up).
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

# ---- weekday-aligned universe ----
idx_all = close.index[close.index.dayofweek < 5]
idx = idx_all[(idx_all >= pd.Timestamp('2020-01-01')) & (idx_all <= pd.Timestamp('2026-07-15'))]
VAL = idx[idx >= pd.Timestamp('2021-01-01')]
SYMS = list(close.columns)
print(f"weekday index: {len(idx)} rows, {idx.min().date()}..{idx.max().date()}; VAL {len(VAL)} rows")

C = close.reindex(idx).astype(float)
O = opn.reindex(idx).astype(float)
H = high.reindex(idx).astype(float)
L = low.reindex(idx).astype(float)
R = ret.reindex(idx).astype(float)
LR = np.log(C / C.shift(1))
VOL20 = R.rolling(20).std()
VOL60 = R.rolling(60).std()

# forward returns
fwd = {}
for h in (1, 2, 3, 5, 10, 20, 30):
    fwd[h] = C.shift(-h) / C - 1.0

# macro signals reindexed to weekdays (ffill)
M = {}
for m in macro.columns:
    s = macro[m].reindex(idx)
    s = s.ffill().astype(float)
    M[m] = s
DVIX = np.log(M['VIX'] / M['VIX'].shift(1))
DDXY = np.log(M['DXY'] / M['DXY'].shift(1))
DJPY = np.log(M['USDJPY'] / M['USDJPY'].shift(1))


def rolling_beta(x, f, win):
    """rolling beta of each asset col vs factor series f (same index)."""
    cov = x.rolling(win).cov(f)
    var = f.rolling(win).var()
    return cov.div(var.replace(0, np.nan), axis=0)


def cs_z(p):
    return p.sub(p.mean(axis=1), axis=0).div(p.std(axis=1).replace(0, np.nan), axis=0)


# ==================== candidate factors ====================
cands = {}

# --- Family A: crash / downside risk (long orientation = safer) ---
cands['skew60_neg'] = -LR.rolling(60, min_periods=30).skew()
dd60 = np.sqrt((LR.clip(lower=0) ** 2).rolling(60, min_periods=30).mean())
cands['downside60_neg'] = -(dd60 / (LR.rolling(60, min_periods=30).std() + 1e-12))
roll_max = C.rolling(60, min_periods=20).max()
cands['maxdd60'] = (C / roll_max - 1.0)  # higher (closer to 0) = less drawdown
cands['var95_neg60'] = -LR.rolling(60, min_periods=30).quantile(0.05).abs()

# --- Family B: trend quality / positioning ---
ma50 = C.rolling(50, min_periods=30).mean()
cands['dist_ma50'] = C / ma50 - 1.0
cands['hh60'] = C / C.rolling(60, min_periods=30).max() - 1.0
er20 = (C / C.shift(20) - 1.0).abs() / (LR.abs().rolling(20).sum() + 1e-12)
cands['er20'] = er20
up20 = R.clip(lower=0).rolling(20).sum()
dn20 = (-R).clip(lower=0).rolling(20).sum()
cands['updown20'] = up20 / (dn20 + 1e-12)

# --- Family C: macro / cross-asset beta ---
cands['vix_beta60'] = rolling_beta(LR, DVIX, 60)
cands['dxy_beta60'] = rolling_beta(LR, DDXY, 60)
cands['usdjpy_beta60'] = rolling_beta(LR, DJPY, 60)
world_ex = (R.sum(axis=1) - R) / (R.notna().sum(axis=1) - 1).replace(0, np.nan)
cands['world_beta60'] = rolling_beta(LR, world_ex, 60)
cands['btc_beta60'] = rolling_beta(LR, np.log(C['BTC'] / C['BTC'].shift(1)), 60)

# --- Family D: volatility dynamics ---
cands['vol_ratio_5_60'] = -(VOL20 / (VOL60 + 1e-12) - 1.0)  # higher = calmer recently
pk20 = np.sqrt((np.log(H / L).pow(2)).rolling(20).mean() / (4 * np.log(2)))
cands['gap_ratio20'] = -(pk20 / (VOL20 + 1e-12) - 1.0)  # higher = fewer gaps
volz20 = (VOL20 - VOL20.rolling(120, min_periods=60).mean()) / (VOL20.rolling(120, min_periods=60).std() + 1e-12)
cands['volz20_mom'] = volz20 - volz20.shift(10)  # vol regime change

# --- Family E: seasonality ---
dow_ret = R.groupby(R.index.dayofweek).transform(lambda s: s.rolling(52 * 5, min_periods=100).mean())
cands['dow_ret52'] = dow_ret
# same-calendar-month average return from past years (exclude current year, trailing)
m_hist = {}
for s in SYMS:
    sr = R[s]
    df = pd.DataFrame({'ret': sr, 'month': sr.index.month, 'year': sr.index.year})
    avg = df.groupby('month')['ret'].transform(lambda g: g.expanding().mean().shift(1))
    m_hist[s] = avg
cands['month_hist'] = pd.DataFrame(m_hist, index=R.index)

# --- Family F: cross-sectional rank of 5d return reversal scaled by trend (novel combo) ---
cands['rev5_x_er20'] = -np.log(C / C.shift(5)) * er20  # reversal when trend inefficient

N_CELLS = len(VAL) * len(SYMS)
GATE_IC, GATE_ICIR = 0.0070, 0.0840

# ==================== library proxies (real signal reconstruction) ====================
lib = {}
lib['mom10_skip5'] = np.log(C / C.shift(10)) - np.log(C / C.shift(5))
lib['mom10_skip5_alt'] = C.shift(5) / C.shift(15) - 1.0
lib['mom120_skip5'] = C.shift(5) / C.shift(125) - 1.0
lib['rev1'] = -R
lib['nclv1'] = -(C - L) / (H - L).replace(0, np.nan)
lib['intraday_rev'] = 1.0 - C / O
lib['vol_of_vol'] = R.rolling(20).std().rolling(60).std()
lib['vix_beta_cond'] = -rolling_beta(LR, DVIX, 60).mul(M['VIX'] / M['VIX'].shift(20) - 1.0, axis=0)
# volz_20 in library is a VOLUME z-score (only crypto+equities have volume); proxy with available cols
volz_vol = (VOL20 - VOL20.rolling(120, min_periods=60).mean()) / (VOL20.rolling(120, min_periods=60).std() + 1e-12)
lib['volz20'] = volz_vol


def panel_corr(a, b):
    A = a.values.astype(float)
    B = b.values.astype(float)
    m = np.isfinite(A) & np.isfinite(B)
    if int(m.sum()) < 50:
        return np.nan
    x = A[m]
    y = B[m]
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


print(f"\n{'='*110}")
res = {}
for nm, p in cands.items():
    try:
        res[nm] = evaluate(nm, p)
    except Exception as e:
        print(f"{nm}: ERROR {e}")

passers = {k: v for k, v in res.items() if v['passed']}
print(f"\nTotal candidates: {len(cands)}, PASS: {len(passers)} -> {list(passers.keys())}")
