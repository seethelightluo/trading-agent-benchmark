"""Round 19 novel factor screen (miner_2). Fresh ideas:
- Crypto beta family: btc_beta_cond_60x20, btc_beta_60, eth_beta_60
- Rates beta family: us10y_beta_60, us10y_beta_cond_60x20, rate_spread_beta_60
- Regional equity betas: hsi_beta_60, n225_beta_60, sx5e_beta_60
- Ratio-spread betas: wti_copper_ratio_beta_20, btc_eth_ratio_beta_20
- Conditional betas: ndx_beta_cond_60x20, xau_beta_cond_60x20
- Relative momentum vs gold/SPX, liquidity-stress (Amihud z-score)

Validates vs the full EFFECTIVE library (signal artifacts).
"""
import sys, time, json
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from pathlib import Path
from factor_common import (load_prices, load_index, factor_to_panel,
                           validate_factor, canonical_grid, signal_matrix,
                           WATCHLIST)

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
print(f'library factors: {len(lib)} {sorted(lib.keys())}', flush=True)

def lib_max_corr(panel):
    arr = signal_matrix(panel, grid)
    best, best_id = 0.0, None
    for fid, larr in lib.items():
        corrs = []
        for i in range(arr.shape[0]):
            x, y = arr[i], larr[i]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                xr = pd.Series(x[m]).rank().values
                yr = pd.Series(y[m]).rank().values
                c = np.corrcoef(xr, yr)[0, 1]
                if np.isfinite(c):
                    corrs.append(c)
        if corrs:
            r = float(np.mean(corrs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id

# ---- benchmark return helpers ----
def bench_ret(sym, as_pct=True):
    df = prices[sym]
    r = df['close'].pct_change()
    return r if not as_pct else r

def rolling_beta(asset_df, bench_ret_series, window, min_obs=30, bench_is_yield=False):
    r = asset_df['close'].pct_change()
    b = bench_ret_series
    if bench_is_yield:
        b = bench_ret_series.diff()
    z = pd.concat([r.rename('r'), b.rename('b')], axis=1).dropna()
    cov = z['r'].rolling(window, min_periods=min_obs).cov(z['b'])
    var = z['b'].rolling(window, min_periods=min_obs).var()
    return (cov / var).reindex(z.index)

def momentum_20(sym):
    df = prices[sym]
    return df['close'] / df['close'].shift(20) - 1.0

# ---- candidate factor functions ----
def f_btc_beta_cond_60x20(df, s):
    if 'BTC' not in prices:
        return None
    b = rolling_beta(df, bench_ret('BTC'), 60)
    return (b * momentum_20('BTC')).reindex(b.index)

def f_btc_beta_60(df, s):
    if 'BTC' not in prices:
        return None
    return rolling_beta(df, bench_ret('BTC'), 60)

def f_eth_beta_60(df, s):
    if 'ETH' not in prices:
        return None
    return rolling_beta(df, bench_ret('ETH'), 60)

def f_us10y_beta_60(df, s):
    if 'US10Y' not in prices:
        return None
    return rolling_beta(df, prices['US10Y']['close'], 60, bench_is_yield=True)

def f_us10y_beta_cond_60x20(df, s):
    if 'US10Y' not in prices:
        return None
    b = rolling_beta(df, prices['US10Y']['close'], 60, bench_is_yield=True)
    move = prices['US10Y']['close'].diff(20)
    return (b * move).reindex(b.index)

def f_rate_spread_beta_60(df, s):
    if 'US10Y' not in prices or 'CN10Y' not in prices:
        return None
    spread = prices['CN10Y']['close'] - prices['US10Y']['close']
    return rolling_beta(df, spread, 60, bench_is_yield=True)

def f_hsi_beta_60(df, s):
    if 'HSI' not in prices:
        return None
    return rolling_beta(df, bench_ret('HSI'), 60)

def f_n225_beta_60(df, s):
    if 'N225' not in prices:
        return None
    return rolling_beta(df, bench_ret('N225'), 60)

def f_sx5e_beta_60(df, s):
    if 'SX5E' not in prices:
        return None
    return rolling_beta(df, bench_ret('SX5E'), 60)

def f_ndx_beta_cond_60x20(df, s):
    if 'NDX' not in prices:
        return None
    b = rolling_beta(df, bench_ret('NDX'), 60)
    return (b * momentum_20('NDX')).reindex(b.index)

def f_xau_beta_cond_60x20(df, s):
    if 'XAU' not in prices:
        return None
    b = rolling_beta(df, bench_ret('XAU'), 60)
    return (b * momentum_20('XAU')).reindex(b.index)

def f_wti_copper_ratio_beta_20(df, s):
    if 'WTI' not in prices or 'COPPER' not in prices:
        return None
    ratio = prices['WTI']['close'] / prices['COPPER']['close']
    return rolling_beta(df, ratio, 20, min_obs=12, bench_is_yield=False)

def f_btc_eth_ratio_beta_20(df, s):
    if 'BTC' not in prices or 'ETH' not in prices:
        return None
    ratio = prices['BTC']['close'] / prices['ETH']['close']
    return rolling_beta(df, ratio, 20, min_obs=12, bench_is_yield=False)

def f_gold_rel_mom_20(df, s):
    if 'XAU' not in prices:
        return None
    mine = df['close'] / df['close'].shift(20) - 1.0
    gold = prices['XAU']['close'] / prices['XAU']['close'].shift(20) - 1.0
    return (mine - gold).reindex(mine.index)

def f_spx_rel_mom_20(df, s):
    if 'SPX' not in prices:
        return None
    mine = df['close'] / df['close'].shift(20) - 1.0
    spx = prices['SPX']['close'] / prices['SPX']['close'].shift(20) - 1.0
    return (mine - spx).reindex(mine.index)

def f_amihud_z_20_60(df, s):
    # per-asset z-score of Amihud illiquidity (|ret|/volume), mean over last 20d
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
    panel = factor_to_panel(fn, prices)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f'{fid}: insufficient data -> None', flush=True)
        results[fid] = {'ok': False, 'metrics': None}
        continue
    rho, rho_id = lib_max_corr(panel)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = rho_id
    ic_ok = abs(m['ic']) >= 0.007
    icir_ok = abs(m['icir']) >= 0.084
    corr_ok = rho < 0.5
    ok = ic_ok and icir_ok and corr_ok
    results[fid] = {'ok': ok, 'metrics': m}
    print(f'Factor {fid}: panel {panel.shape} range {panel.index.min()}..{panel.index.max()}', flush=True)
    print(json.dumps({k: v for k, v in m.items() if k != 'decay_ic_by_horizon'}, indent=2, default=str), flush=True)
    print('decay:', json.dumps(m['decay_ic_by_horizon'], default=str), flush=True)
    print(f"ADMISSION: |IC|={abs(m['ic']):.4f}>=0.007 {ic_ok} | |ICIR|={abs(m['icir']):.4f}>=0.084 {icir_ok} | corr={rho:.3f}<0.5 {corr_ok} -> {'PASS' if ok else 'FAIL'}", flush=True)
    print('-' * 80, flush=True)

out = Path('scripts/miner_2_20260730_results_round19.json')
out.write_text(json.dumps(results, indent=1, default=str), encoding='utf-8')
print(f'elapsed={time.time()-t0:.1f}s results -> {out}', flush=True)
