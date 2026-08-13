"""miner_2 2033-02-03: explore a batch of novel factor candidates (one idea each).

Admission battery (shared contract):
  - warm-up window 2020-01-01..2026-07-15, h=10
  - |IC| >= 0.0070 and |ICIR| >= 0.0840
  - redundancy audit vs ALL 22 persisted library signal artifacts (canonical grid)
  - OOS robustness window 2026-07-16..latest visible date reported for context
"""
import sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, load_index, factor_to_panel,
                           validate_factor, canonical_grid, signal_matrix,
                           forward_returns, rank_ic_series, VAL_START, VAL_END)

np.seterr(all='ignore')
t0 = time.time()
prices = load_prices(days=3500)
max_date = max(dd.index.max() for dd in prices.values())
print(f"data max_date={max_date.date()} n_assets={len(prices)} ({time.time()-t0:.1f}s)", flush=True)

spx = prices['SPX']['close']
ndx = prices['NDX']['close']
wti = prices['WTI']['close']
xau = prices['XAU']['close']
grid = canonical_grid(prices)
print(f"canonical grid: {len(grid)} dates {grid.min().date()}..{grid.max().date()}", flush=True)

# ---------------- market references ----------------
def beta_to(mkt_r, w=60, minp=40):
    def _f(df, s):
        r = df['close'].pct_change()
        z = pd.concat([r.rename('r'), mkt_r.rename('m')], axis=1).dropna()
        b = z['r'].rolling(w, min_periods=minp).cov(z['m']) / z['m'].rolling(w, min_periods=minp).var().replace(0, np.nan)
        return b.reindex(df.index)
    return _f

# ---------------- candidate factor definitions ----------------
CAND = {}

def f_drawdown_depth_120(df, s):
    """Current drawdown depth from 120d running max (negative)."""
    c = df['close']
    return (c / c.rolling(120, min_periods=60).max() - 1.0)
CAND['drawdown_depth_120'] = (f_drawdown_depth_120, "drawdown depth vs 120d max", "close/rolling_max(close,120)-1", ["close"], {"window": 120})

def f_downside_vol_ratio_60(df, s):
    """Downside semi-vol / total vol over 60d (risk asymmetry)."""
    r = df['close'].pct_change()
    dn = r.where(r < 0)
    dv = dn.rolling(60, min_periods=40).std()
    tv = r.rolling(60, min_periods=40).std()
    return (dv / tv.replace(0, np.nan)).reindex(df.index)
CAND['downside_vol_ratio_60'] = (f_downside_vol_ratio_60, "downside semi-vol / total vol 60d", "STD60(min(ret,0))/STD60(ret)", ["close"], {"window": 60})

def f_vol_term_5_60(df, s):
    """Short-term (5d) vol vs long (60d) vol; spikes when >1."""
    r = df['close'].pct_change()
    v5 = r.rolling(5, min_periods=4).std()
    v60 = r.rolling(60, min_periods=40).std()
    return (v5 / v60.replace(0, np.nan)).reindex(df.index)
CAND['vol_term_5_60'] = (f_vol_term_5_60, "vol term structure 5d/60d", "STD5(ret)/STD60(ret)", ["close"], {"short": 5, "long": 60})

def f_bollinger_pos_20(df, s):
    """Bollinger position: (close-SMA20)/(2*STD20)."""
    c = df['close']
    ma = c.rolling(20, min_periods=15).mean()
    sd = c.rolling(20, min_periods=15).std()
    return ((c - ma) / (2 * sd.replace(0, np.nan))).reindex(df.index)
CAND['bollinger_pos_20'] = (f_bollinger_pos_20, "Bollinger band position 20d", "(close-SMA20)/(2*STD20)", ["close"], {"window": 20})

def f_kurt_60(df, s):
    """Realized kurtosis of daily returns over 60d (tail risk)."""
    r = df['close'].pct_change()
    return r.rolling(60, min_periods=40).kurt()
CAND['kurt_60'] = (f_kurt_60, "realized kurtosis 60d", "KURT60(ret)", ["close"], {"window": 60})

def f_autocorr_60(df, s):
    """Lag-1 autocorrelation of daily returns over 60d (magnitude, not sign)."""
    r = df['close'].pct_change()
    return r.rolling(60, min_periods=40).apply(lambda x: pd.Series(x).autocorr(1) if len(x) >= 40 else np.nan, raw=False)
CAND['autocorr_60'] = (f_autocorr_60, "return lag-1 autocorrelation 60d", "AC1(ret,60)", ["close"], {"window": 60})

def f_overnight_share_20(df, s):
    """Overnight gap contribution to returns over 20d."""
    c = df['close']
    on = c / df['open'] - 1.0  # overnight (prev close -> open is in open/prev close; approximate)
    tot = c.pct_change()
    a_on = on.rolling(20, min_periods=12).mean().abs()
    a_tot = tot.rolling(20, min_periods=12).mean().abs()
    return (a_on / a_tot.replace(0, np.nan)).reindex(df.index)
CAND['overnight_share_20'] = (f_overnight_share_20, "overnight gap share of return 20d", "|mean(overnight)|/|mean(total)| 20d", ["close", "open"], {"window": 20})

def f_gap_size_20(df, s):
    """Mean absolute gap size (open/prev_close-1) over 20d (gap_freq measures count)."""
    g = (df['open'] / df['close'].shift(1) - 1.0).abs()
    return g.rolling(20, min_periods=12).mean()
CAND['gap_size_20'] = (f_gap_size_20, "mean absolute gap size 20d", "mean(|open/prev_close-1|,20)", ["open", "close"], {"window": 20})

def f_volume_trend_20_60(df, s):
    """Participation expansion: 20d avg volume / 60d avg volume."""
    v = df['volume']
    return (v.rolling(20, min_periods=12).mean() / v.rolling(60, min_periods=30).mean().replace(0, np.nan)).reindex(df.index)
CAND['volume_trend_20_60'] = (f_volume_trend_20_60, "volume trend 20/60", "MA20(vol)/MA60(vol)", ["volume"], {"short": 20, "long": 60})

def f_up_vol_ratio_20(df, s):
    """Up-day volume vs down-day volume over 20d (buying pressure)."""
    r = df['close'].pct_change()
    v = df['volume']
    up = v.where(r > 0).rolling(20, min_periods=8).mean()
    dn = v.where(r < 0).rolling(20, min_periods=8).mean()
    return (up / dn.replace(0, np.nan)).reindex(df.index)
CAND['up_vol_ratio_20'] = (f_up_vol_ratio_20, "up/down volume ratio 20d", "MA20(vol|ret>0)/MA20(vol|ret<0)", ["close", "volume"], {"window": 20})

def f_skew_60(df, s):
    """Realized skewness of close-to-close returns over 60d."""
    r = df['close'].pct_change()
    return r.rolling(60, min_periods=40).skew()
CAND['skew_60'] = (f_skew_60, "realized skewness 60d", "SKEW60(ret)", ["close"], {"window": 60})

CAND['wti_beta_60'] = (beta_to(wti.pct_change()), "beta to WTI 60d", "BETA60(ret,WTI_ret)", ["close"], {"window": 60, "mkt": "WTI"})
CAND['ndx_beta_60'] = (beta_to(ndx.pct_change()), "beta to NDX 60d (tech linkage)", "BETA60(ret,NDX_ret)", ["close"], {"window": 60, "mkt": "NDX"})
CAND['xau_beta_60'] = (beta_to(xau.pct_change()), "beta to XAU 60d (gold linkage)", "BETA60(ret,XAU_ret)", ["close"], {"window": 60, "mkt": "XAU"})

# ---------------- library redundancy audit ----------------
lib_artifacts = []
for p in sorted(Path('factors').glob('*_signal.npy')):
    fid = p.name.replace('_signal.npy', '')
    arr = np.load(p, allow_pickle=False)
    if arr.shape == (len(grid), 15):
        lib_artifacts.append((fid, arr))
print(f"library artifacts for rho audit: {len(lib_artifacts)}", flush=True)

def max_lib_rho(panel):
    m = signal_matrix(panel, grid)
    best, best_id = 0.0, None
    for fid, arr in lib_artifacts:
        corrs = []
        for i in range(m.shape[0]):
            x, y = m[i], arr[i]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 8:
                c = pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank())
                if np.isfinite(c):
                    corrs.append(c)
        if corrs:
            r = float(np.mean(corrs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id

# ---------------- validation ----------------
H = 10
fwd_all = {h: forward_returns(prices, h) for h in (1, 2, 3, 5, 10, 20)}
oos_start = VAL_END + pd.Timedelta(days=1)
recent_start = max_date - pd.Timedelta(days=365)
FROZEN = {'HSI', 'SX5E', 'BTC', 'US10Y', 'CN10Y'}
LIVE = [s for s in WATCHLIST if s not in FROZEN]

print(f"\n{'factor':24s} {'warm_ic':>8s} {'warm_icir':>9s} {'hit':>5s} {'cov':>5s} {'turn':>6s} {'rho':>5s} {'oos_ic':>8s} {'recent_ic':>9s} {'recent_icir':>11s} {'gate':>4s}", flush=True)
results = {}
for name, (fn, desc, expr, deps, params) in CAND.items():
    t1 = time.time()
    panel = factor_to_panel(fn, prices)
    m = validate_factor(name, panel, prices)
    if m is None:
        print(f"{name:24s} insufficient data", flush=True)
        continue
    rho, fid = max_lib_rho(panel)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = fid
    # OOS / recent context
    oos_p = panel[(panel.index >= oos_start)]
    oos_f = fwd_all[H].reindex(oos_p.index)
    oos_ic = rank_ic_series(oos_p[LIVE], oos_f[LIVE], min_valid=8)
    rec_p = panel[(panel.index >= recent_start)]
    rec_f = fwd_all[H].reindex(rec_p.index)
    rec_ic = rank_ic_series(rec_p[LIVE], rec_f[LIVE], min_valid=8)
    rec_icir = rec_ic.mean() / rec_ic.std() * np.sqrt(len(rec_ic)) if len(rec_ic) > 1 and rec_ic.std() > 0 else np.nan
    oos_ic_mean = float(oos_ic.mean()) if len(oos_ic) else np.nan
    rec_ic_mean = float(rec_ic.mean()) if len(rec_ic) else np.nan
    gate = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
    results[name] = dict(m, oos_ic=oos_ic_mean, recent_ic=rec_ic_mean, recent_icir=rec_icir,
                         recent_n=int(rec_ic.notna().sum()), oos_n=int(oos_ic.notna().sum()))
    print(f"{name:24s} {m['ic']:+8.4f} {m['icir']:+9.3f} {m['ic_hit_ratio']:5.2f} {m['coverage_asset_days']:5.2f} "
          f"{m['turnover_10d_rank']:6.2f} {rho:5.2f} {oos_ic_mean:+8.4f} {rec_ic_mean:+9.4f} {rec_icir:+11.3f} {'PASS' if gate else 'FAIL'}"
          f"  ({time.time()-t1:.1f}s)", flush=True)

with open('scripts/miner_2_20330203_explore_results.json', 'w') as f:
    json.dump(results, f, indent=1, default=str)
print("\nsaved scripts/miner_2_20330203_explore_results.json", flush=True)
