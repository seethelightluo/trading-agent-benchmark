"""Batch B (2027-01-14) miner_2: novel price/OHLC-only factor screen.

Universe: 15 cross-asset tradable instruments (no volume reliance: SOX/XAU/COPPER/
WTI/US10Y/CN10Y have degenerate volume).

Validation on warm-up 2020-01-01..2026-07-15 canonical grid.
Admission gate: |IC10| >= 0.007 AND |ICIR10| >= 0.084; supplementary online
window IC (2026-07-16..data end) reported separately. max_abs_library_correlation
reported vs real signal artifacts of EFFECTIVE library factors.

One idea per factor_fn; candidates are intentionally in untested signal space
(no overlap with miner_1 batch A: updown_vol_ratio_20, spx_hsi_ratio_beta_60,
boll_bandwidth_20, downside_beta_20, zscore_20).
"""
import sys, json, glob, time
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, load_index, canonical_grid,
                           signal_matrix, VAL_START, VAL_END, forward_returns,
                           factor_to_panel, validate_factor)

np.seterr(all='ignore')
t0 = time.time()
prices = load_prices(days=2000)
grid = canonical_grid(prices)
T, N = len(grid), len(WATCHLIST)
val_mask = (grid >= VAL_START) & (grid <= VAL_END)
print(f"grid {T} dates {grid.min().date()}..{grid.max().date()} | assets {len(prices)} | {time.time()-t0:.1f}s", flush=True)
print(f"full data end: {max(d.index.max() for d in prices.values()).date()}", flush=True)

# ---------------- library rank matrices from real signal artifacts ----------------
lib_raw = {}
for f in sorted(glob.glob('factors/*.json')):
    try:
        d = json.load(open(f))
        if d.get('validation', {}).get('status') != 'EFFECTIVE':
            continue
        fid = d['factor_id']
        art = d.get('signal_artifact')
        if not art:
            continue
        arr = np.load('factors/' + art, allow_pickle=False)
        if arr.shape == (T, N):
            lib_raw[fid] = arr
    except Exception:
        pass
print(f"library artifacts: {len(lib_raw)} -> {sorted(lib_raw.keys())}", flush=True)


def rank_matrix(df):
    return df.rank(axis=1).values.astype(float)


def spearman_from_ranks(xr, yr):
    valid = np.isfinite(xr) & np.isfinite(yr)
    nv = valid.sum(axis=1)
    ok = nv >= 8
    out = np.full(len(nv), np.nan)
    xc = np.where(valid, xr, np.nan); yc = np.where(valid, yr, np.nan)
    mx = np.nanmean(xc, axis=1, keepdims=True); my = np.nanmean(yc, axis=1, keepdims=True)
    xc = np.where(valid, xr - mx, 0.0); yc = np.where(valid, yr - my, 0.0)
    num = (xc * yc).sum(axis=1)
    den = np.sqrt((xc * xc).sum(axis=1) * (yc * yc).sum(axis=1))
    out[ok] = num[ok] / den[ok]
    return out


def max_lib_rho_from_artifacts(panel):
    """Mean daily cross-sectional Spearman vs each effective library artifact."""
    best = 0.0; best_id = None
    xr = rank_matrix(panel.reindex(grid))
    for fid, arr in lib_raw.items():
        yr = rank_matrix(pd.DataFrame(arr, index=grid, columns=WATCHLIST))
        corrs = spearman_from_ranks(xr, yr)
        r = float(np.nanmean(corrs)) if np.isfinite(corrs).any() else 0.0
        if abs(r) > best:
            best = abs(r); best_id = fid
    return best, best_id


fwd_raw = {h: forward_returns(prices, h) for h in (1, 2, 3, 5, 10, 20)}
fwd_rank = {h: rank_matrix(fwd_raw[h].reindex(grid)) for h in fwd_raw}
lib_rank = {fid: rank_matrix(pd.DataFrame(arr, index=grid, columns=WATCHLIST)) for fid, arr in lib_raw.items()}

# online supplementary grid
all_dates = sorted(set().union(*[set(d.index) for d in prices.values()]))
grid2 = pd.DatetimeIndex([d for d in all_dates if d > VAL_END])
fwd10_online = forward_returns(prices, 10).reindex(grid2)
fwd10_online_rank = rank_matrix(fwd10_online)
print(f"online grid: {len(grid2)} dates {grid2.min().date()}..{grid2.max().date()}", flush=True)


def ic10_series(panel):
    fac = panel.reindex(grid)
    fr = fwd_rank[10]
    xr = rank_matrix(fac)
    return spearman_from_ranks(xr, fr)


def online_metrics(panel):
    fac = panel.reindex(grid2)
    xr = rank_matrix(fac)
    ic = spearman_from_ranks(xr, fwd10_online_rank)
    ic = ic[np.isfinite(ic)]
    if len(ic) < 20:
        return None
    m = float(np.mean(ic)); s = float(np.std(ic, ddof=1))
    return {'online_ic': m, 'online_icir': m / s if s > 0 else 0.0, 'online_n': int(len(ic))}


# ---------------- candidate definitions ----------------
def f_park_ratio_20(df, s):
    """Parkinson intraday-range vol / close-close vol (20d). High = more info in intraday range."""
    c = df['close']; h = df['high']; l = df['low']
    r = c.pct_change()
    park = (np.log(h / l) ** 2 / (4 * np.log(2))).rolling(20).mean() ** 0.5
    cc = r.rolling(20).std()
    return (park / cc).replace([np.inf, -np.inf], np.nan)


def f_gap_intraday_20(df, s):
    """Overnight gap return minus intraday open-close return (20d mean)."""
    o = df['open']; c = df['close']; pc = df['close'].shift(1)
    gap = o / pc - 1.0
    intra = c / o - 1.0
    return (gap - intra).rolling(20).mean()


def f_ret_autocorr_20(df, s):
    """Lag-1 autocorrelation of daily returns over 20d (persistence of daily direction)."""
    r = df['close'].pct_change()
    return r.rolling(20).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if np.isfinite(x).all() and np.std(x[:-1]) > 0 and np.std(x[1:]) > 0 else np.nan, raw=False)


def f_vol_asym_20(df, s):
    """Downside vol / upside vol (20d): high = fatter downside moves."""
    r = df['close'].pct_change()
    up = r.where(r > 0, 0.0)
    dn = r.where(r < 0, 0.0)
    uvol = (up ** 2).rolling(20).mean() ** 0.5
    dvol = (dn ** 2).rolling(20).mean() ** 0.5
    return (dvol / uvol).replace([np.inf, -np.inf], np.nan)


def f_drawup_dd_60(df, s):
    """60d max drawup duration - max drawdown duration (normalized by days)."""
    c = df['close']
    def _dur(ser):
        peak = ser.cummax(); dd_days = (ser < peak).astype(float)
        # duration of longest uninterrupted drawdown spell
        best = cur = 0.0
        for v in dd_days.values:
            cur = cur + 1 if v else 0.0
            best = max(best, cur)
        return best
    roll = c.rolling(60)
    dd_dur = roll.apply(lambda x: _dur(pd.Series(x)), raw=False)
    # drawup = inverse series (trough to peak)
    inv = -c
    du_dur = inv.rolling(60).apply(lambda x: _dur(pd.Series(x)), raw=False)
    return (du_dur - dd_dur)


def f_global_basket_beta_60(df, s, basket=None):
    """Rolling 60d beta of asset vs equal-weight cross-asset basket return."""
    if basket is None:
        return None
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), basket.reindex(r.index).rename('b')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['b']) / z['b'].rolling(60).var()
    return b


def f_range_expansion_20x60(df, s):
    """Range expansion: 20d high-low range / 60d mean range."""
    h = df['high']; l = df['low']
    rng20 = (h - l).rolling(20).mean()
    rng60 = (h - l).rolling(60).mean()
    return (rng20 / rng60).replace([np.inf, -np.inf], np.nan)


def f_trend_adherence_60(df, s):
    """Trend adherence: fraction of days where close moves in direction of 60d drift."""
    c = df['close']
    drift = c / c.shift(60) - 1.0
    ret = c.pct_change()
    aligned = np.sign(drift) * np.sign(ret)
    return aligned.rolling(60).mean()


# build global basket (equal-weight cross-asset mean return; alignment handled per asset)
basket_ret = None
{
    (lambda: None)()
}
ret_panels = []
for s, df in prices.items():
    ret_panels.append(df['close'].pct_change().rename(s))
basket = pd.concat(ret_panels, axis=1).mean(axis=1).sort_index()
basket = basket[basket.index <= grid.max()]

candidates = {
    'park_ratio_20': (f_park_ratio_20, 'Parkinson intraday-range vol / close-close vol (20d)', 'intraday range vs realized vol'),
    'gap_intraday_20': (f_gap_intraday_20, 'overnight gap return - intraday open-close return (20d mean)', 'overnight/intraday asymmetry'),
    'ret_autocorr_20': (f_ret_autocorr_20, 'lag-1 autocorrelation of daily returns (20d)', 'daily return persistence'),
    'vol_asym_20': (f_vol_asym_20, 'downside vol / upside vol (20d)', 'vol asymmetry / crash sensitivity'),
    'drawup_dd_60': (f_drawup_dd_60, '60d max drawup duration - max drawdown duration', 'trend vs chop balance'),
    'global_basket_beta_60': (f_global_basket_beta_60, 'rolling 60d beta vs equal-weight cross-asset basket', 'global systematic beta'),
    'range_expansion_20x60': (f_range_expansion_20x60, '20d mean range / 60d mean range', 'range expansion/contraction'),
    'trend_adherence_60': (f_trend_adherence_60, 'fraction of daily moves aligned with 60d drift', 'trend consistency'),
}

results = {}
for fid, (fn, desc, tag) in candidates.items():
    try:
        if fid == 'global_basket_beta_60':
            panel = factor_to_panel(lambda df, s: f_global_basket_beta_60(df, s, basket=basket), prices)
        else:
            panel = factor_to_panel(fn, prices)
        m = validate_factor(fid, panel, prices)
        if m is None:
            print(f"{fid}: insufficient data", flush=True)
            results[fid] = {'ok': False, 'desc': desc, 'tag': tag, 'metrics': None}
            continue
        rho, fid_max = max_lib_rho_from_artifacts(panel)
        m['max_abs_library_correlation'] = rho
        m['max_corr_library_id'] = fid_max
        om = online_metrics(panel)
        if om:
            m.update(om)
        ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
        results[fid] = {'ok': ok, 'desc': desc, 'tag': tag, 'metrics': m}
        print(f"\n=== {fid} ({tag}) ===", flush=True)
        print(f"  IC10={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} n={m['n_ic_dates']} "
              f"cov={m['coverage_asset_days']:.3f} cov8={m['coverage_dates_ge8']:.3f} turn={m['turnover_10d_rank']:.2f} "
              f"rho_lib={rho:.3f}({fid_max})", flush=True)
        if om:
            print(f"  ONLINE: IC={om['online_ic']:+.4f} ICIR={om['online_icir']:+.4f} n={om['online_n']}", flush=True)
        print(f"  decay: " + ", ".join(f"{h}:{m['decay_ic_by_horizon'][h]:+.4f}" for h in ('1','2','3','5','10','20')), flush=True)
        print(f"  ADMISSION: {'PASS' if ok else 'FAIL'}", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        results[fid] = {'ok': False, 'desc': desc, 'tag': tag, 'metrics': None, 'error': str(e)}

with open('scripts/miner_2_20270114_results_batchB.json', 'w') as fh:
    json.dump(results, fh, indent=1, default=str)
print("\n=== SUMMARY ===", flush=True)
for fid, v in results.items():
    m = v.get('metrics') or {}
    print(f"{fid:28s} ok={v['ok']} ic={m.get('ic', float('nan')):+.4f} icir={m.get('icir', float('nan')):+.4f} "
          f"rho={m.get('max_abs_library_correlation', float('nan')):.3f} online_ic={m.get('online_ic', float('nan')):+.4f}", flush=True)
print(f"total {time.time()-t0:.1f}s", flush=True)
