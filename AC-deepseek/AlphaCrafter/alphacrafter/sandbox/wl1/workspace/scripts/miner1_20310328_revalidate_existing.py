"""miner1 2031-03-28: re-validate all 14 currently effective factors through 2031-03-27.
Full-sample (2021-01-01..) + recent windows; gates |IC1|>=0.007, |ICIR1|>=0.084 (1d horizon)."""
import pandas as pd, numpy as np, json

panel = pd.read_pickle('scripts/panel_cache_20310328.pkl')
close, high, low, open_, vol, macro = (panel['close'], panel['high'], panel['low'],
                                       panel['open'], panel['vol'], panel['macro'])
ret = close.pct_change()
lnc = np.log(close)

def daily_ic(factor_df, fwd, min_valid=8):
    out = {}
    for dt in factor_df.index:
        f = factor_df.loc[dt]; r = fwd.loc[dt]
        m = f.notna() & r.notna()
        if m.sum() < min_valid:
            continue
        ic = f[m].rank().corr(r[m].rank())
        if np.isfinite(ic):
            out[dt] = ic
    return pd.Series(out, dtype=float)

def summ(ic_s):
    ic = ic_s.dropna()
    if len(ic) == 0:
        return None
    mean = ic.mean(); std = ic.std(ddof=1)
    return {'n': int(len(ic)), 'ic': float(mean),
            'icir': float(mean/std) if std > 0 else np.nan,
            'hit': float((ic > 0).mean()), 'ic_std': float(std) if std > 0 else np.nan}

def cov_turn(fdf):
    cov = float(fdf.notna().mean(axis=1).mean())
    ranks = fdf.rank(axis=1) / fdf.notna().sum(axis=1)
    turn = float(ranks.diff().abs().mean().mean())
    return cov, turn

# ---- factor definitions ----
factors = {}
factors['miner2_20260715_rev_1d'] = -(lnc.diff(1))
factors['miner2_20260715_rev_2d'] = -(lnc.diff(2))
factors['miner2_20260715_rev_3d'] = -(lnc.diff(3))
factors['miner2_20260715_rev_5d'] = -(lnc.diff(5))
factors['miner2_20260715_nclv_1d'] = -(close - low.rolling(1).min()) / (high.rolling(1).max() - low.rolling(1).min())
factors['miner2_20260715_nclv_2d'] = -(close - low.rolling(2).min()) / (high.rolling(2).max() - low.rolling(2).min())
factors['miner2_20260715_nclv_3d'] = -(close - low.rolling(3).min()) / (high.rolling(3).max() - low.rolling(3).min())
factors['miner2_20260715_nclv_5d'] = -(close - low.rolling(5).min()) / (high.rolling(5).max() - low.rolling(5).min())
factors['miner2_20260715_nbody_1d'] = -(close - open_) / (high - low)
factors['miner2_20260715_id_rev_1d'] = -(close / open_ - 1.0)
factors['miner2_20260715_rev_1d_vs'] = -(lnc.diff(1)) / ret.rolling(20).std()
factors['mom_120d_skip5'] = close.shift(5) / close.shift(125) - 1.0
vix = macro['VIX']
vixr = vix.pct_change()
def roll_beta(x, y, w=60):
    return x.rolling(w).cov(y) / y.rolling(w).var()
beta_vix = roll_beta(ret, vixr, 60)
factors['vix_beta_cond_60x20'] = -beta_vix * (vix / vix.shift(20) - 1.0)
factors['vol_of_vol20x60'] = ret.rolling(20).std().rolling(60).std()

# forward returns
fwd = {h: close.shift(-h) / close - 1.0 for h in (1, 2, 5, 10)}

windows = {
    'full': (pd.Timestamp('2021-01-01'), pd.Timestamp('2031-03-27')),
    'recent_2y': (pd.Timestamp('2029-04-01'), pd.Timestamp('2031-03-27')),
    'recent_1y': (pd.Timestamp('2030-04-01'), pd.Timestamp('2031-03-27')),
}

results = {}
for fid, fdf in factors.items():
    fdf = fdf.reindex(close.index)
    results[fid] = {}
    for wname, (a, b) in windows.items():
        sub = fdf[(fdf.index >= a) & (fdf.index <= b)]
        h1 = fwd[1].reindex(sub.index)
        ic1 = daily_ic(sub, h1)
        s = summ(ic1)
        cov, turn = cov_turn(sub)
        results[fid][wname] = {'ic1': s, 'cov': cov, 'turn': turn}

# print summary
gate_ic, gate_icir = 0.007, 0.084
print(f"{'factor':34s} | {'full IC/ICIR':>16s} | {'2y IC/ICIR':>16s} | {'1y IC/ICIR':>16s} | 1y gate")
for fid in factors:
    r = results[fid]
    def fmt(w):
        s = r[w]['ic1']
        if s is None: return '  n/a        '
        return f"{s['ic']:+.4f}/{s['icir']:+.3f}"
    f_full, f_2y, f_1y = fmt('full'), fmt('recent_2y'), fmt('recent_1y')
    s1y = r['recent_1y']['ic1']
    ok = 'PASS' if (s1y is not None and abs(s1y['ic']) >= gate_ic and abs(s1y['icir']) >= gate_icir) else 'fail'
    print(f"{fid:34s} | {f_full:>16s} | {f_2y:>16s} | {f_1y:>16s} | {ok}")

print("\n--- detail: full-sample 1d IC stats ---")
for fid in factors:
    s = results[fid]['full']['ic1']
    if s:
        print(f"{fid:34s} n={s['n']:5d} IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['hit']:.3f} cov={results[fid]['full']['cov']:.3f} turn={results[fid]['full']['turn']:.3f}")

with open('scripts/miner1_20310328_reval_existing.json', 'w') as f:
    json.dump(results, f, indent=1, default=str)
print("\nsaved scripts/miner1_20310328_reval_existing.json")
