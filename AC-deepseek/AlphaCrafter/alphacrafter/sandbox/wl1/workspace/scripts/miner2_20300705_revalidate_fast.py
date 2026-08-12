"""miner2 2030-07-05: fast vectorized re-validation of existing miner2 factors on fresh panel (through 2030-07-04)."""
import pandas as pd, numpy as np, sys, json
sys.path.insert(0, 'scripts')

P = pd.read_pickle('scripts/panel_cache_20300705.pkl')
close, high, low, open_, ret = P['close'], P['high'], P['low'], P['open'], P['ret']


def fwd_ret(close, h):
    return close.shift(-h) / close - 1.0


def fast_rank_ic(signal, fwd, min_n=8):
    """Vectorized row-wise Spearman rank IC (Pearson on ranks)."""
    sig_rank = signal.rank(axis=1)
    fwd_rank = fwd.rank(axis=1)
    mask = signal.notna() & fwd.notna()
    n = mask.sum(axis=1)
    valid = n >= min_n
    sx = sig_rank.where(mask)
    sy = fwd_rank.where(mask)
    mx = sx.sum(axis=1) / n.replace(0, np.nan)
    my = sy.sum(axis=1) / n.replace(0, np.nan)
    xc = sx.sub(mx, axis=0).where(mask)
    yc = sy.sub(my, axis=0).where(mask)
    num = (xc * yc).sum(axis=1)
    den = np.sqrt((xc ** 2).sum(axis=1) * (yc ** 2).sum(axis=1))
    ic = num / den
    ic = ic[valid]
    return ic.values, signal.index[valid]


def eval_factor(signal, close, horizons=(1, 2, 3, 5, 10), min_n=8, start=None, end=None):
    if start is not None:
        signal = signal[signal.index >= start]
    if end is not None:
        signal = signal[signal.index <= end]
    out = {}
    for h in horizons:
        fwd = fwd_ret(close, h)
        ics, dates = fast_rank_ic(signal, fwd, min_n=min_n)
        if len(ics) == 0:
            out[h] = dict(ic=np.nan, icir=np.nan, hit=np.nan, n=0)
            continue
        ic = float(np.mean(ics))
        sd = float(np.std(ics, ddof=1))
        icir = ic / sd if sd > 0 else np.nan
        hit = float(np.mean(ics > 0))
        out[h] = dict(ic=ic, icir=icir, hit=hit, n=len(ics))
    cov = float(signal.notna().mean().mean()) if signal.shape[0] else 0.0
    rp = signal.rank(axis=1, pct=True)
    to = float(rp.diff().abs().mean().mean()) if rp.shape[0] > 1 else 0.0
    out['coverage'] = cov
    out['turnover_1d_rank'] = to
    out['n_dates'] = int(signal.shape[0])
    return out


def summarize(res, label=''):
    h1 = res.get(1, {})
    h5 = res.get(5, {})
    print(f"[{label}] IC1={h1.get('ic', float('nan')):.4f} ICIR1={h1.get('icir', float('nan')):.3f} "
          f"hit1={h1.get('hit', float('nan')):.3f} n1={h1.get('n', 0)} | "
          f"IC5={h5.get('ic', float('nan')):.4f} ICIR5={h5.get('icir', float('nan')):.3f} | "
          f"cov={res.get('coverage', float('nan')):.3f} turn1d={res.get('turnover_1d_rank', float('nan')):.3f} "
          f"dates={res.get('n_dates', 0)}")
    return h1


factors = {}
for nd in [1, 2, 3, 5]:
    factors[f'rev_{nd}d'] = -(np.log(close) - np.log(close.shift(nd)))
for nd in [1, 2, 3, 5]:
    rng = high.rolling(nd).max() - low.rolling(nd).min()
    factors[f'nclv_{nd}d'] = -(close - low.rolling(nd).min()) / rng.replace(0, np.nan)
factors['id_rev_1d'] = -(close / open_ - 1.0)
factors['nbody_1d'] = -(close - open_) / (high - low).replace(0, np.nan)
factors['rev_1d_vs'] = -(np.log(close) - np.log(close.shift(1))) / ret.rolling(20).std().replace(0, np.nan)

FULL_START, FULL_END = '2021-01-01', '2030-07-04'
RECENT_START = '2028-01-01'
RECENT2_START = '2029-06-01'

print("=" * 112)
print(f"RE-VALIDATION of existing miner2 factors | FULL {FULL_START}..{FULL_END} | RECENT {RECENT_START}+ | R2029+ {RECENT2_START}+")
print("=" * 112)
full_res, recent_res, recent2_res = {}, {}, {}
for name, sig in factors.items():
    r_full = eval_factor(sig, close, start=FULL_START, end=FULL_END)
    r_recent = eval_factor(sig, close, start=RECENT_START, end=FULL_END)
    r_recent2 = eval_factor(sig, close, start=RECENT2_START, end=FULL_END)
    full_res[name] = r_full
    recent_res[name] = r_recent
    recent2_res[name] = r_recent2
    summarize(r_full, f"FULL  {name}")
    summarize(r_recent, f"RECENT {name}")
    summarize(r_recent2, f"R2029+ {name}")

print("\nYear-by-year IC1 (full sample):")
for name in ['rev_1d', 'rev_2d', 'nclv_1d', 'nclv_2d', 'id_rev_1d', 'nbody_1d']:
    sig = factors[name]
    yrs = {}
    for y in range(2021, 2031):
        r = eval_factor(sig, close, start=f'{y}-01-01', end=f'{y}-12-31')
        yrs[str(y)] = r.get(1, {})
        h1 = r.get(1, {})
        print(f"  {name} {y}: IC1={h1.get('ic', float('nan')):.4f} ICIR1={h1.get('icir', float('nan')):.3f} n={h1.get('n', 0)}")
    full_res[name]['by_year_ic1'] = yrs

with open('scripts/miner2_reval_20300705.json', 'w') as f:
    json.dump({'full': {k: {str(ik): iv for ik, iv in v.items()} for k, v in full_res.items()},
               'recent': {k: {str(ik): iv for ik, iv in v.items()} for k, v in recent_res.items()},
               'recent2': {k: {str(ik): iv for ik, iv in v.items()} for k, v in recent2_res.items()}}, f, indent=1, default=str)
print("\nsaved scripts/miner2_reval_20300705.json")
