"""
miner1 2028-06-09: focused validation of trend-conditioned momentum family.
Admission convention follows existing library (mom_120d_skip5.json used
admission_horizon=10): report IC/ICIR at h=1..10, sub-period robustness at h=10,
coverage, turnover. Library-correlation vs existing artifacts computed for best.
"""
import pandas as pd, numpy as np, sys, json, base64, zlib, hashlib
sys.path.insert(0, 'scripts')
from miner1_ic_lib import load_panel, ic_series, fwd_returns, summarize_ic, coverage_stats, turnover_signal, WATCH

panel = load_panel()
close = panel['close']
fw = fwd_returns(close, horizons=(1, 2, 3, 5, 10))


def mom_skip(px, n, skip):
    return px.shift(skip) / px.shift(skip + n) - 1.0


def sma(px, n):
    return px.rolling(n).mean()


m120 = mom_skip(close, 120, 5)
m60 = mom_skip(close, 60, 5)
sma20, sma60, sma100, sma120, sma200 = (sma(close, n) for n in [20, 60, 100, 120, 200])

cands = {
    'mom120_x_tr60':  m120 * (close > sma60).astype(float),
    'mom120_x_tr100': m120 * (close > sma100).astype(float),
    'mom120_x_tr120': m120 * (close > sma120).astype(float),
    'mom120_x_tr200': m120 * (close > sma200).astype(float),
    'mom120_x_tr60x20': m120 * ((close > sma60) & (close > sma20)).astype(float),
    'mom60_x_tr120':  m60 * (close > sma120).astype(float),
    'mom120_skip5_BASELINE': m120,
}

print("=== horizon sweep (2021-01-01 .. 2028-06-08) ===")
print(f"{'variant':24s} {'h':>2s} {'IC':>8s} {'ICIR':>8s} {'hit':>6s} {'n':>5s}")
for name, f in cands.items():
    f = f.loc['2021-01-01':]
    for h in [1, 2, 3, 5, 10]:
        ic = ic_series(f, fw[h])
        s = summarize_ic(ic, f'{name} h{h}')
        if s:
            print(f"{name:24s} {h:2d} {s['mean_ic']:+8.4f} {s['icir']:+8.4f} {s['hit_rate']:6.3f} {s['n_dates']:5d}")

print("\n=== sub-period robustness at h=10 ===")
for name in ['mom120_x_tr60', 'mom120_x_tr100', 'mom120_x_tr120', 'mom120_x_tr60x20', 'mom120_skip5_BASELINE']:
    f = cands[name]
    print(name)
    for lo, hi in [('2021-01-01', '2022-12-31'), ('2023-01-01', '2024-12-31'),
                   ('2025-01-01', '2026-07-15'), ('2026-07-16', '2027-12-31'),
                   ('2028-01-01', '2028-06-08')]:
        ff = f.loc[lo:hi]
        ic = ic_series(ff, fw[10].loc[lo:hi])
        s = summarize_ic(ic, f'{lo}..{hi}')
        if s:
            print(f"  {lo}..{hi} IC={s['mean_ic']:+.4f} ICIR={s['icir']:+.4f} hit={s['hit_rate']:.3f} n={s['n_dates']}")

print("\n=== coverage & turnover (h=1 window) ===")
for name in ['mom120_x_tr60', 'mom120_x_tr100', 'mom120_x_tr120', 'mom120_x_tr60x20', 'mom120_skip5_BASELINE']:
    f = cands[name].loc['2021-01-01':]
    cov = coverage_stats(f)
    print(f"{name:24s} covD_ge8={cov['dates_valid_ge8']:5d}/{cov['total_dates']:5d} avgV={cov['avg_valid']:5.1f} "
          f"rank_to={turnover_signal(f):.4f}")

print("\n=== library correlation for mom120_x_tr60 (daily cross-sectional Pearson vs library artifacts) ===")
lib_ids = ['mom_120d_skip5', 'vol_of_vol20x60', 'vix_beta_cond_60x20',
           'miner2_20260715_nclv_1d', 'miner2_20260715_rev_2d', 'miner2_20260715_rev_1d']
sig = cands['mom120_x_tr60']


def load_lib_signal(fid):
    d = json.load(open(f'factors/{fid}.json'))
    art = d.get('validation', {}).get('signal_artifact', {})
    data = art.get('data')
    if not data:
        return None
    raw = zlib.decompress(base64.b64decode(data)).decode()
    df = pd.read_csv(pd.io.common.StringIO(raw), index_col=0, parse_dates=True)
    return df


cand_aligned = sig
for fid in lib_ids:
    try:
        lib = load_lib_signal(fid)
    except Exception as e:
        print(f'  {fid}: cannot load ({e})')
        continue
    if lib is None:
        print(f'  {fid}: no artifact')
        continue
    common = cand_aligned.index.intersection(lib.index)
    if len(common) < 100:
        print(f'  {fid}: only {len(common)} common dates, skip')
        continue
    a = cand_aligned.loc[common]
    b = lib.loc[common]
    rhos = []
    for t in common:
        x = a.loc[t].astype(float)
        y = b.loc[t].astype(float)
        m = x.notna() & y.notna()
        if m.sum() >= 5:
            r = np.corrcoef(x[m], y[m])[0, 1]
            if np.isfinite(r):
                rhos.append(r)
    if len(rhos) > 50:
        print(f'  {fid}: mean_rho={np.mean(rhos):+.4f} mean_abs_rho={np.mean(np.abs(rhos)):.4f} n={len(rhos)}')
    else:
        print(f'  {fid}: insufficient pairs ({len(rhos)})')
