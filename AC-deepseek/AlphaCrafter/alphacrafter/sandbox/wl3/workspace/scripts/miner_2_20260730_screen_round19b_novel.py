"""Round 19b novel factor screen (miner_2) - VECTORIZED lib-corr, fast.
Fresh ideas:
- Crypto beta family: btc_beta_cond_60x20, btc_beta_60, eth_beta_60
- Rates beta family: us10y_beta_60, us10y_beta_cond_60x20, rate_spread_beta_60
- Regional equity betas: hsi_beta_60, n225_beta_60, sx5e_beta_60
- Ratio-spread betas: wti_copper_ratio_beta_20, btc_eth_ratio_beta_20
- Conditional betas: ndx_beta_cond_60x20, xau_beta_cond_60x20
- Relative momentum vs gold/SPX, liquidity-stress (Amihud z-score)
Validates vs the full EFFECTIVE library (signal artifacts), vectorized.
"""
import sys, time, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from pathlib import Path
from factor_common import (load_prices, factor_to_panel, validate_factor,
                           canonical_grid, signal_matrix, WATCHLIST)

t0 = time.time()
prices = load_prices(days=2000)
grid = canonical_grid(prices)
print(f'prices={len(prices)} grid={len(grid)} {grid.min().date()}..{grid.max().date()}', flush=True)

# ---- library: effective factors with signal artifacts ----
lib = {}
for p in sorted(Path('factors').glob('*.json')):
    if p.name.endswith('.bak') or 'deprecated' in p.name or 'ensemble' in p.name:
        continue
    try:
        payload = json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        continue
    if payload.get('validation', {}).get('status') != 'EFFECTIVE':
        continue
    art = payload.get('signal_artifact')
    art_path = p.parent / str(art) if art else None
    if art_path is not None and art_path.exists():
        lib[payload['factor_id']] = np.load(art_path, allow_pickle=False)
print(f'library factors: {len(lib)}', flush=True)

def rank_matrix(arr):
    """Row-wise Spearman ranks -> (T,15), NaN where invalid. Vectorized per row."""
    out = np.full(arr.shape, np.nan)
    T = arr.shape[0]
    for t in range(T):
        row = arr[t]
        m = np.isfinite(row)
        n = m.sum()
        if n >= 8:
            r = np.full(15, np.nan)
            r[m] = (pd.Series(row[m]).rank().values)
            out[t] = r
    return out

# precompute library rank matrices once
lib_rank = {fid: rank_matrix(arr) for fid, arr in lib.items()}

def lib_max_corr_fast(panel):
    arr = signal_matrix(panel, grid)
    cand_rank = rank_matrix(arr)
    best, best_id = 0.0, None
    for fid, lr in lib_rank.items():
        cs = np.full(cand_rank.shape[0], np.nan)
        for t in range(cand_rank.shape[0]):
            a, b = cand_rank[t], lr[t]
            m = np.isfinite(a) & np.isfinite(b)
            n = m.sum()
            if n >= 8:
                a2, b2 = a[m], b[m]
                ma, mb = a2.mean(), b2.mean()
                sa, sb = a2.std(ddof=1), b2.std(ddof=1)
                if sa > 1e-12 and sb > 1e-12:
                    cs[t] = ((a2 - ma) * (b2 - mb)).mean() / (sa * sb)
        if np.isfinite(cs).any():
            r = float(np.nanmean(cs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id

# ---- benchmark return helpers ----
def bench_ret(sym):
    return prices[sym]['close'].pct_change()

def rolling_beta(asset_df, bench_series, window, min_obs=30, bench_is_yield=False):
    r = asset_df['close'].pct_change()
    b = bench_series.diff() if bench_is_yield else bench_series
    z = pd.concat([r.rename('r'), b.rename('b')], axis=1).dropna()
    cov = z['r'].rolling(window, min_periods=min_obs).cov(z['b'])
    var = z['b'].rolling(window, min_periods=min_obs).var()
    return (cov / var).reindex(z.index)

def momentum_20(sym):
    df = prices[sym]
    return df['close'] / df['close'].shift(20) - 1.0

# ---- candidate factor functions ----
def f_btc_beta_cond_60x20(df, s):
    b = rolling_beta(df, bench_ret('BTC'), 60)
    return (b * momentum_20('BTC')).reindex(b.index)

def f_btc_beta_60(df, s):
    return rolling_beta(df, bench_ret('BTC'), 60)

def f_eth_beta_60(df, s):
    return rolling_beta(df, bench_ret('ETH'), 60)

def f_us10y_beta_60(df, s):
    return rolling_beta(df, prices['US10Y']['close'], 60, bench_is_yield=True)

def f_us10y_beta_cond_60x20(df, s):
    b = rolling_beta(df, prices['US10Y']['close'], 60, bench_is_yield=True)
    move = prices['US10Y']['close'].diff(20)
    return (b * move).reindex(b.index)

def f_rate_spread_beta_60(df, s):
    spread = prices['CN10Y']['close'] - prices['US10Y']['close']
    return rolling_beta(df, spread, 60, bench_is_yield=True)

def f_hsi_beta_60(df, s):
    return rolling_beta(df, bench_ret('HSI'), 60)

def f_n225_beta_60(df, s):
    return rolling_beta(df, bench_ret('N225'), 60)

def f_sx5e_beta_60(df, s):
    return rolling_beta(df, bench_ret('SX5E'), 60)

def f_ndx_beta_cond_60x20(df, s):
    b = rolling_beta(df, bench_ret('NDX'), 60)
    return (b * momentum_20('NDX')).reindex(b.index)

def f_xau_beta_cond_60x20(df, s):
    b = rolling_beta(df, bench_ret('XAU'), 60)
    return (b * momentum_20('XAU')).reindex(b.index)

def f_wti_copper_ratio_beta_20(df, s):
    ratio = prices['WTI']['close'] / prices['COPPER']['close']
    return rolling_beta(df, ratio, 20, min_obs=12)

def f_btc_eth_ratio_beta_20(df, s):
    ratio = prices['BTC']['close'] / prices['ETH']['close']
    return rolling_beta(df, ratio, 20, min_obs=12)

def f_gold_rel_mom_20(df, s):
    mine = df['close'] / df['close'].shift(20) - 1.0
    gold = prices['XAU']['close'] / prices['XAU']['close'].shift(20) - 1.0
    return (mine - gold).reindex(mine.index)

def f_spx_rel_mom_20(df, s):
    mine = df['close'] / df['close'].shift(20) - 1.0
    spx = prices['SPX']['close'] / prices['SPX']['close'].shift(20) - 1.0
    return (mine - spx).reindex(mine.index)

def f_amihud_z_20_60(df, s):
    r = df['close'].pct_change().abs()
    v = df['volume'].astype(float).replace(0, np.nan)
    ami = (r / v).rolling(60, min_periods=30).mean()
    mu = ami.rolling(60, min_periods=30).mean()
    sd = ami.rolling(60, min_periods=30).std()
    z = (ami - mu) / sd
    return z.rolling(20, min_periods=10).mean()

CANDIDATES = [
    ('btc_beta_cond_60x20', f_btc_beta_cond_60x20),
    ('btc_beta_60', f_btc_beta_60),
    ('eth_beta_60', f_eth_beta_60),
    ('us10y_beta_60', f_us10y_beta_60),
    ('us10y_beta_cond_60x20', f_us10y_beta_cond_60x20),
    ('rate_spread_beta_60', f_rate_spread_beta_60),
    ('hsi_beta_60', f_hsi_beta_60),
    ('n225_beta_60', f_n225_beta_60),
    ('sx5e_beta_60', f_sx5e_beta_60),
    ('ndx_beta_cond_60x20', f_ndx_beta_cond_60x20),
    ('xau_beta_cond_60x20', f_xau_beta_cond_60x20),
    ('wti_copper_ratio_beta_20', f_wti_copper_ratio_beta_20),
    ('btc_eth_ratio_beta_20', f_btc_eth_ratio_beta_20),
    ('gold_rel_mom_20', f_gold_rel_mom_20),
    ('spx_rel_mom_20', f_spx_rel_mom_20),
    ('amihud_z_20_60', f_amihud_z_20_60),
]

results = {}
for fid, fn in CANDIDATES:
    t1 = time.time()
    panel = factor_to_panel(fn, prices)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f'{fid}: insufficient data -> None', flush=True)
        results[fid] = {'ok': False, 'metrics': None}
        continue
    rho, rho_id = lib_max_corr_fast(panel)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = rho_id
    ic_ok = abs(m['ic']) >= 0.007
    icir_ok = abs(m['icir']) >= 0.084
    corr_ok = rho < 0.5
    ok = ic_ok and icir_ok and corr_ok
    results[fid] = {'ok': ok, 'metrics': m}
    print(f'Factor {fid}: panel {panel.shape} [{time.time()-t1:.1f}s]', flush=True)
    print(json.dumps({k: v for k, v in m.items() if k != 'decay_ic_by_horizon'}, indent=2, default=str), flush=True)
    print('decay:', json.dumps(m['decay_ic_by_horizon'], default=str), flush=True)
    print(f"ADMISSION: |IC|={abs(m['ic']):.4f}>=0.007 {ic_ok} | |ICIR|={abs(m['icir']):.4f}>=0.084 {icir_ok} | corr={rho:.3f}<0.5 {corr_ok} -> {'PASS' if ok else 'FAIL'}", flush=True)
    print('-' * 80, flush=True)

out = Path('scripts/miner_2_20260730_results_round19.json')
out.write_text(json.dumps(results, indent=1, default=str), encoding='utf-8')
print(f'elapsed={time.time()-t0:.1f}s results -> {out}', flush=True)
