"""miner2 2031-01-03: re-validate existing miner2 + library factors on fresh panel (through 2031-01-02)."""
import pandas as pd, numpy as np, sys, json
sys.path.insert(0, 'scripts')

P = pd.read_pickle('scripts/panel_cache_20310103.pkl')
close, high, low, open_, ret = P['close'], P['high'], P['low'], P['open'], P['ret']
END = str(close.index.max().date())

# --- macro alignment fix: reindex macro onto asset panel index ---
macro = P['macro'].reindex(close.index)


def fwd_ret(close, h):
    return close.shift(-h) / close - 1.0


def fast_rank_ic(signal, fwd, min_n=8):
    fwd = fwd.reindex(signal.index)
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
    h10 = res.get(10, {})
    print(f"[{label}] IC1={h1.get('ic', float('nan')):.4f} ICIR1={h1.get('icir', float('nan')):.3f} "
          f"hit1={h1.get('hit', float('nan')):.3f} n1={h1.get('n', 0)} | "
          f"IC5={h5.get('ic', float('nan')):.4f} ICIR5={h5.get('icir', float('nan')):.3f} | "
          f"IC10={h10.get('ic', float('nan')):.4f} ICIR10={h10.get('icir', float('nan')):.3f} | "
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

# library factors
factors['mom_120d_skip5'] = close / close.shift(120) - 1.0
vol20 = ret.rolling(20).std()
vol60 = ret.rolling(60).std()
factors['vol_of_vol20x60'] = vol20 / vol60 - 1.0
vix = macro['VIX']
vix_ret = vix.pct_change()
asset_ret = ret
beta60 = asset_ret.rolling(60).cov(vix_ret) / vix_ret.rolling(60).var()
vix_norm = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
factors['vix_beta_cond_60x20'] = -beta60 * (vix_norm > 0.5).astype(float)

FULL_START, FULL_END = '2021-01-01', END
RECENT_START = '2028-01-01'
RECENT2_START = '2030-06-01'

print("=" * 130)
print(f"RE-VALIDATION | FULL {FULL_START}..{FULL_END} | RECENT {RECENT_START}+ | R2030H2+ {RECENT2_START}+")

results = {'full': {}, 'recent': {}, 'recent2': {}}
for name, sig in factors.items():
    for tag, (st, en) in {'full': (FULL_START, FULL_END), 'recent': (RECENT_START, FULL_END),
                          'recent2': (RECENT2_START, FULL_END)}.items():
        results[tag][name] = eval_factor(sig, close, start=st, end=en)

for tag in ['full', 'recent', 'recent2']:
    print(f"\n--- {tag.upper()} ---")
    for name in factors:
        summarize(results[tag][name], name)

with open('scripts/miner2_reval_20310103.json', 'w') as f:
    json.dump(results, f, indent=1, default=str)
print("\nsaved scripts/miner2_reval_20310103.json")

# ---- library correlation (signal-level pairwise rho, daily cross-section avg) ----
print("\n" + "=" * 130)
print("LIBRARY SIGNAL CORRELATION (avg pairwise rho per date, 2021+)")
sigs = {k: v for k, v in factors.items()}
names = list(sigs.keys())
rho_mat = pd.DataFrame(index=names, columns=names, dtype=float)
for i in names:
    for j in names:
        a = sigs[i].rank(axis=1)
        b = sigs[j].rank(axis=1)
        mask = a.notna() & b.notna()
        n = mask.sum(axis=1)
        valid = n >= 8
        rho_day = a.where(mask).corrwith(b.where(mask), axis=1)
        rho_mat.loc[i, j] = float(rho_day[valid].mean())
print(rho_mat.round(3))
max_abs = rho_mat.abs().values
np.fill_diagonal(max_abs, np.nan)
print("\nmax |rho| non-diag:", np.nanmax(max_abs))
