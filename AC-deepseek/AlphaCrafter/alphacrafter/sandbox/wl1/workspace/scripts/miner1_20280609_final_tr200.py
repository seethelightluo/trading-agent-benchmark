"""
miner1 2028-06-09: final check for mom120_x_tr200 (SMA200 gate) vs mom120_x_tr60.
Pick the stronger, more robust variant for persistence.
"""
import pandas as pd, numpy as np, sys, json, base64, zlib
sys.path.insert(0, 'scripts')
from miner1_ic_lib import load_panel, ic_series, fwd_returns, summarize_ic, coverage_stats, turnover_signal

panel = load_panel()
close = panel['close']
fw = fwd_returns(close, horizons=(1, 2, 3, 5, 10))


def mom_skip(px, n, skip):
    return px.shift(skip) / px.shift(skip + n) - 1.0


def sma(px, n):
    return px.rolling(n).mean()


m120 = mom_skip(close, 120, 5)
cands = {
    'mom120_x_tr60': m120 * (close > sma(close, 60)).astype(float),
    'mom120_x_tr200': m120 * (close > sma(close, 200)).astype(float),
}

print("=== h=10 sub-periods ===")
for name, f in cands.items():
    print(name)
    for lo, hi in [('2021-01-01', '2022-12-31'), ('2023-01-01', '2024-12-31'),
                   ('2025-01-01', '2026-07-15'), ('2026-07-16', '2027-12-31'),
                   ('2028-01-01', '2028-06-08')]:
        ff = f.loc[lo:hi]
        ic = ic_series(ff, fw[10].loc[lo:hi])
        s = summarize_ic(ic, f'{lo}..{hi}')
        if s:
            print(f"  {lo}..{hi} IC={s['mean_ic']:+.4f} ICIR={s['icir']:+.4f} hit={s['hit_rate']:.3f} n={s['n_dates']}")

print("\n=== turnover (h=1 window) ===")
for name, f in cands.items():
    print(f"{name:20s} signal_turnover={turnover_signal(f.loc['2021-01-01':]):.4f}")

print("\n=== library correlation for mom120_x_tr200 ===")
sig = cands['mom120_x_tr200']
lib_ids = ['mom_120d_skip5', 'vol_of_vol20x60', 'vix_beta_cond_60x20']


def load_lib_signal(fid):
    d = json.load(open(f'factors/{fid}.json'))
    art = d.get('validation', {}).get('signal_artifact', {})
    data = art.get('data')
    if not data:
        return None
    raw = zlib.decompress(base64.b64decode(data)).decode()
    return pd.read_csv(pd.io.common.StringIO(raw), index_col=0, parse_dates=True)


for fid in lib_ids:
    lib = load_lib_signal(fid)
    if lib is None:
        continue
    common = sig.index.intersection(lib.index)
    rhos = []
    for t in common:
        x = sig.loc[t].astype(float)
        y = lib.loc[t].astype(float)
        m = x.notna() & y.notna()
        if m.sum() >= 5:
            r = np.corrcoef(x[m], y[m])[0, 1]
            if np.isfinite(r):
                rhos.append(r)
    print(f"  {fid}: mean_abs_rho={np.mean(np.abs(rhos)):.4f} n={len(rhos)}")

# correlation tr60 vs tr200
common = cands['mom120_x_tr60'].index.intersection(cands['mom120_x_tr200'].index)
rhos = []
for t in common:
    x = cands['mom120_x_tr60'].loc[t].astype(float)
    y = cands['mom120_x_tr200'].loc[t].astype(float)
    m = x.notna() & y.notna()
    if m.sum() >= 5:
        r = np.corrcoef(x[m], y[m])[0, 1]
        if np.isfinite(r):
            rhos.append(r)
print(f"  tr60 vs tr200: mean_abs_rho={np.mean(np.abs(rhos)):.4f} n={len(rhos)}")
