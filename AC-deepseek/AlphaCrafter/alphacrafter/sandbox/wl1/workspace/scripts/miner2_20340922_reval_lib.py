"""miner_2 2034-09-22: revalidate the full factor library on the fresh panel (data through 2034-09-21)."""
import numpy as np
import pandas as pd
import json, glob, os

with open('scripts/panel_cache_20340922.pkl', 'rb') as f:
    P = pd.read_pickle(f)
close = P['close']; high = P['high']; low = P['low']; opn = P['open']; vol = P['vol']; ret = P['ret']; macro = P['macro']

def ln(x):
    return np.log(x)

def sig(x):
    return np.sign(x)

def corr_pair(a, b):
    m = a.notna() & b.notna()
    if m.sum() < 30:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])

def factor_signal(fid):
    """Return the raw signal DataFrame (index=date, cols=assets) for a library factor. Direction handled by caller."""
    d = json.load(open(f'factors/{fid}.json'))
    expr = d['calculation']['expression']
    params = d.get('parameters', {})
    # simple evaluator for known library expressions
    if fid.startswith('miner2_20260715_rev_'):
        if fid.endswith('_vs'):
            nd = 1
            s = -(ln(close) - ln(close.shift(nd)))
            # volume-scaled: multiply by sign of volume deviation
            vmean = vol.rolling(20).mean().replace(0, np.nan)
            vs = vol / vmean
            s = s * np.sign(vs - 1.0)
        else:
            nd = int(fid.split('_')[-1].replace('d', ''))
            s = -(ln(close) - ln(close.shift(nd)))
        return s
    if fid.startswith('miner2_20260715_nclv_'):
        nd = int(fid.split('_')[-1].replace('d', ''))
        rnd = ret.rolling(nd).mean()
        vnd = ret.rolling(nd).std()
        s = -(rnd / vnd.replace(0, np.nan))
        return s
    if fid.startswith('miner2_20260715_nbody_'):
        body = (close - opn) / opn
        rng = (high - low) / close
        s = -(body / rng.replace(0, np.nan))
        return s
    if fid.startswith('miner2_20260715_id_rev_'):
        nd = 1
        gap = opn / close.shift(1) - 1.0
        s = -gap
        return s
    if fid == 'mom_120d_skip5':
        s = ln(close) - ln(close.shift(120))
        # skip5: drop last 5 days
        s = s - (ln(close) - ln(close.shift(5)))
        return s
    if fid == 'vol_of_vol20x60':
        v20 = ret.rolling(20).std()
        v60 = ret.rolling(60).std()
        s = v20 / v60 - 1.0
        return s
    if fid == 'vix_beta_cond_60x20':
        vix = macro['VIX']
        dvix = vix.pct_change()
        # beta of asset returns on dVIX over 60d, conditional on |dVIX|>threshold, sign-adjusted by VIX level trend over 20d
        s = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
        for c in close.columns:
            r = ret[c]
            df = pd.concat([r, dvix], axis=1).dropna()
            if len(df) < 100:
                continue
            # rolling 60d beta
            rb = r.rolling(60).corr(dvix) * (r.rolling(60).std() / dvix.rolling(60).std())
            s[c] = rb
        return s
    raise ValueError('unknown factor ' + fid)

def rank_ic(sig, fwd):
    """Daily cross-sectional rank IC of signal vs forward return."""
    fr = ret.shift(-fwd)
    dates, ics = [], []
    for dt in sig.index:
        s = sig.loc[dt]
        f = fr.loc[dt]
        m = s.notna() & f.notna() & np.isfinite(s) & np.isfinite(f)
        if m.sum() >= 8:
            ic = float(np.corrcoef(s[m].rank(), f[m].rank())[0, 1])
            if np.isfinite(ic):
                dates.append(dt); ics.append(ic)
    return pd.Series(ics, index=pd.DatetimeIndex(dates))

def report(fid, direction):
    sig = factor_signal(fid)
    sig = sig.replace([np.inf, -np.inf], np.nan)
    sig = sig * direction
    out = {'factor_id': fid, 'direction': direction}
    for h in [1, 2, 3, 5, 10]:
        ics = rank_ic(sig, h)
        if len(ics) == 0:
            out[f'ic{h}'] = np.nan; out[f'icir{h}'] = np.nan; out[f'n{h}'] = 0
            continue
        ic = float(ics.mean())
        icir = float(ics.mean() / ics.std() * np.sqrt(len(ics))) if ics.std() > 0 else np.nan
        out[f'ic{h}'] = ic; out[f'icir{h}'] = icir; out[f'n{h}'] = len(ics)
        # recent windows on h=1
        for label, wd in [('2y', 504), ('1y', 252), ('6m', 126)]:
            sub = ics.tail(wd)
            if len(sub) >= 60:
                out[f'ic{h}_{label}'] = float(sub.mean())
                out[f'icir{h}_{label}'] = float(sub.mean() / sub.std() * np.sqrt(len(sub))) if sub.std() > 0 else np.nan
            else:
                out[f'ic{h}_{label}'] = np.nan; out[f'icir{h}_{label}'] = np.nan
    return out

fids = [os.path.basename(f)[:-5] for f in glob.glob('factors/*.json') if '.bak' not in f and 'evicted' not in f]
# direction mapping from ensemble or factor files
rows = []
for fid in sorted(fids):
    try:
        d = json.load(open(f'factors/{fid}.json'))
    except Exception as e:
        print('skip', fid, e); continue
    # default direction +1; vix_beta dir -1 per ensemble
    direction = -1 if 'vix_beta' in fid else 1
    r = report(fid, direction)
    rows.append(r)
    print(f"{fid} dir={direction:+d} | ic1 {r['ic1']:.4f}/{r['icir1']:.3f} (n{r['n1']}) 1y {r.get('ic1_1y', np.nan):.4f}/{r.get('icir1_1y', np.nan):.3f} 6m {r.get('ic1_6m', np.nan):.4f}/{r.get('icir1_6m', np.nan):.3f} | ic5 {r['ic5']:.4f}/{r['icir5']:.3f} | ic10 {r['ic10']:.4f}/{r['icir10']:.3f}")

df = pd.DataFrame(rows)
df.to_csv('scripts/miner2_reval_20340922_lib.csv', index=False)
print('\nsaved scripts/miner2_reval_20340922_lib.csv')
