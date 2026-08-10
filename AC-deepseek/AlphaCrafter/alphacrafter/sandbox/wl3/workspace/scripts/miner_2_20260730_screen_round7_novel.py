"""miner_2 2026-07-30 round-7 factor screen (fast, vectorized).

Candidates:
  round6 reruns (fast variants):
   kurt_term_20_60, ovn_gap_mom_20, intraday_share_20, rel_str_xau_60,
   gain_loss_ratio_60, atr_ratio_20_60, ndx_beta_60
  new round7:
   usdjpy_beta_cond_60x20   beta to USDJPY * USDJPY 20d mom (carry/risk proxy)
   usdcny_beta_cond_60x20   beta to USDCNY * USDCNY 20d mom
   dd_depth_60              (close/rolling_max(close,60)-1) drawdown depth
   ovn_var_share_20         overnight variance share var(ovn)/var(total)
   skew_60d                 skewness of 60d returns
   downside_vol_ratio_60    std(neg ret)/std(pos ret)
   eff_ratio_60             |close-close60|/sum|ret| (60d efficiency ratio)

Gate: |IC|>=0.007, |ICIR|>=0.084 (h=10, 2020-01-01..2026-07-15),
rho_vs_library < 0.5.
"""
import sys, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, load_index, canonical_grid,
                           forward_returns, WATCHLIST, VAL_START, VAL_END,
                           signal_matrix)

t0 = time.time()
prices = load_prices(days=2600)
dxy = load_index('DXY', prices=prices)
eurusd = load_index('EURUSD', prices=prices)
usdjpy = load_index('USDJPY', prices=prices)
usdcny = load_index('USDCNY', prices=prices)
vix = load_index('VIX', prices=prices)
grid = canonical_grid(prices)
print(f'[load] {len(prices)} assets; grid {len(grid)} dates '
      f'{grid.min().date()}..{grid.max().date()}', flush=True)

fwd = {h: forward_returns(prices, h) for h in (1, 2, 3, 5, 10, 20)}


def rank_ic_series(panel, fr, min_valid=8):
    common = panel.index.intersection(fr.index)
    ic = {}
    for d in common:
        x, y = panel.loc[d], fr.loc[d]
        m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if m.sum() >= min_valid:
            ic[d] = x[m].rank().corr(y[m].rank())
    return pd.Series(ic).sort_index()


def validate(panel):
    ic_s = {h: rank_ic_series(panel, fwd[h]) for h in fwd}
    ic10 = ic_s[10]
    full = ic10[(ic10.index >= VAL_START) & (ic10.index <= VAL_END)]
    m = float(full.mean()); sd = float(full.std(ddof=1))
    icir = m / sd if sd > 0 else 0.0
    hit = float((full > 0).mean()) if m >= 0 else float((full < 0).mean())
    fac = panel[(panel.index >= VAL_START) & (panel.index <= VAL_END)]
    cov = float(fac.notna().sum().sum()) / (fac.shape[0] * fac.shape[1])
    ge8 = float((fac.notna().sum(axis=1) >= 8).mean())
    ranked = fac.rank(axis=1)
    turn = float(ranked.diff(10).abs().mean().mean()) if len(ranked) > 10 else float('nan')
    decay = {str(h): (float(ic_s[h].mean()) if len(ic_s[h]) else float('nan')) for h in fwd}
    return dict(ic=m, icir=icir, ic_hit_ratio=hit, n=int(len(full)), cov=cov, ge8=ge8,
                turn=turn, decay=decay)


def pairwise_rho_vs_library(panel, lib_panels):
    best, best_id = 0.0, None
    for fid, lp in lib_panels.items():
        idx = panel.index.intersection(lp.index)
        corrs = []
        for d in idx:
            x, y = panel.loc[d], lp.loc[d]
            m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                c = x[m].rank().corr(y[m].rank())
                if np.isfinite(c):
                    corrs.append(c)
        if corrs:
            r = float(np.mean(corrs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id


# ---------------- library panels (12 active) ----------------
def _beta_anchor(anchor_close):
    def f(df, s):
        a = anchor_close.reindex(df.index)
        r = df['close'].pct_change()
        z = pd.concat([r.rename('r'), a.rename('a')], axis=1).dropna()
        b = z['r'].rolling(60).cov(z['a']) / z['a'].rolling(60).var()
        return b.reindex(df.index)
    return f


def _beta_cond(anchor_close, sign=1.0):
    def f(df, s):
        a = anchor_close.reindex(df.index)
        r = df['close'].pct_change()
        z = pd.concat([r.rename('r'), a.rename('a')], axis=1).dropna()
        b = z['r'].rolling(60).cov(z['a']) / z['a'].rolling(60).var()
        mom = (anchor_close / anchor_close.shift(20) - 1.0).reindex(df.index)
        return (sign * b * mom).reindex(df.index)
    return f


def f_hilo(df, s, w=60):
    hi = df['high'].rolling(w).max(); lo = df['low'].rolling(w).min()
    return (df['close'] - lo) / (hi - lo).replace(0, np.nan)


def f_vol_adj_mom(df, s):
    m = df['close'].shift(5) / df['close'].shift(25) - 1.0
    v = df['close'].pct_change().rolling(60).std()
    return m / v.replace(0, np.nan)


def f_max_ret(df, s):
    return df['close'].pct_change().rolling(20).max()


def f_skew_term(df, s):
    r = df['close'].pct_change()
    return r.rolling(20).skew() - r.rolling(60).skew()


def f_mom120(df, s):
    return df['close'].shift(5) / df['close'].shift(125) - 1.0


def f_dd_duration_panel(prices_):
    since = {}
    for s, df in prices_.items():
        c = df['close'].values.astype(float)
        n = len(c)
        rmax = pd.Series(c).rolling(120, min_periods=60).max().values
        out = np.full(n, np.nan)
        for t in range(n):
            if not np.isfinite(rmax[t]):
                continue
            j = t
            while j >= 0 and (not np.isfinite(c[j]) or c[j] < rmax[t]):
                j -= 1
            out[t] = np.log1p(t - j)
        since[s] = pd.Series(out, index=df.index)
    p = pd.DataFrame(since)
    mom = pd.DataFrame({s: f_mom120(df, s) for s, df in prices_.items()})
    z = mom.sub(mom.mean(axis=1), axis=0).div(mom.std(axis=1).replace(0, np.nan), axis=0)
    resid = p.copy()
    for d in p.index:
        y = p.loc[d]; x = z.loc[d]
        m = y.notna() & x.notna() & np.isfinite(y) & np.isfinite(x)
        if m.sum() >= 8:
            xv, yv = x[m].values, y[m].values
            var = float(xv.var())
            if var > 1e-14:
                beta = float(np.cov(xv, yv)[0, 1]) / var
                resid.loc[d, m] = y[m] - beta * x[m]
    return resid


LIB_DEFS = {
    'spx_beta_60':           (lambda df, s: _beta_anchor(prices['SPX']['close'])(df, s), None),
    'vol_adj_mom_20_60':     (f_vol_adj_mom, None),
    'dxy_beta_cond_60x20':   (lambda df, s: _beta_cond(dxy['close'])(df, s), None),
    'hs300_beta_60':         (lambda df, s: _beta_anchor(prices['000300.SH']['close'])(df, s), None),
    'vol_of_vol20x60':       (lambda df, s: df['close'].pct_change().rolling(20).std().rolling(60).std(), None),
    'dd_duration_120_resid': (None, None),
    'vix_beta_cond_60x20':   (lambda df, s: _beta_cond(vix['close'], sign=-1.0)(df, s), None),
    'skew_term_20_60':       (f_skew_term, None),
    'hilo_pos_60':           (lambda df, s: f_hilo(df, s, 60), None),
    'max_ret_20d':           (f_max_ret, None),
    'eurusd_beta_cond_60x20':(lambda df, s: _beta_cond(eurusd['close'])(df, s), None),
    'down_beta_60':          (lambda df, s: _beta_anchor(prices['SPX']['close'])(df, s), None),  # placeholder; replaced below
}

# down_beta_60: beta on SPX-down days only
def f_down_beta(df, s):
    a = prices['SPX']['close'].reindex(df.index)
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), a.rename('a')], axis=1).dropna()
    down = z[z['a'] < 0]
    b = down['r'].rolling(60).cov(down['a']) / down['a'].rolling(60).var()
    return b.reindex(df.index)

lib_panels = {}
for fid, (fn, _) in LIB_DEFS.items():
    if fid == 'dd_duration_120_resid':
        lib_panels[fid] = f_dd_duration_panel(prices)
    elif fid == 'down_beta_60':
        lib_panels[fid] = pd.DataFrame({s: f_down_beta(prices[s], s) for s in WATCHLIST})
    else:
        lib_panels[fid] = pd.DataFrame({s: fn(prices[s], s) for s in WATCHLIST})
    lib_panels[fid] = lib_panels[fid][~lib_panels[fid].index.duplicated(keep='last')].sort_index()
print(f'[lib] rebuilt 12 library panels on canonical grid ({time.time()-t0:.1f}s)', flush=True)

# ---------------- candidates ----------------
def f_kurt_term(df, s):
    r = df['close'].pct_change()
    return r.rolling(20).kurt() - r.rolling(60).kurt()

def f_ovn_gap_mom(df, s):
    ovn = df['open'] / df['close'].shift(1) - 1.0
    return ovn.rolling(20).mean()

def f_intraday_share(df, s):
    ovn = df['open'] / df['close'].shift(1) - 1.0
    intr = df['close'] / df['open'] - 1.0
    return intr.rolling(20).mean() - ovn.rolling(20).mean()

def f_rel_str_xau(df, s):
    xau = prices['XAU']['close'].reindex(df.index)
    return (df['close'] / df['close'].shift(60) - 1.0) - (xau / xau.shift(60) - 1.0)

def f_gain_loss_ratio(df, s):
    r = df['close'].pct_change()
    pos = r.clip(lower=0.0).rolling(60).mean()
    neg = (-r.clip(upper=0.0)).rolling(60).mean()
    return (pos / neg.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

def f_atr_ratio(df, s):
    pc = df['close'].shift(1)
    tr = pd.concat([(df['high'] - df['low']), (df['high'] - pc).abs(),
                    (df['low'] - pc).abs()], axis=1).max(axis=1)
    return (tr.rolling(20).mean() / tr.rolling(60).mean() - 1.0)

def f_ndx_beta(df, s):
    return _beta_anchor(prices['NDX']['close'])(df, s)

def f_usdjpy_beta_cond(df, s):
    return _beta_cond(usdjpy['close'])(df, s)

def f_usdcny_beta_cond(df, s):
    return _beta_cond(usdcny['close'])(df, s)

def f_dd_depth(df, s):
    return df['close'] / df['close'].rolling(60).max() - 1.0

def f_ovn_var_share(df, s):
    ovn = df['open'] / df['close'].shift(1) - 1.0
    intr = df['close'] / df['open'] - 1.0
    tot = df['close'].pct_change()
    return ovn.rolling(20).var() / tot.rolling(20).var().replace(0, np.nan)

def f_skew60(df, s):
    return df['close'].pct_change().rolling(60).skew()

def f_downside_vol_ratio(df, s):
    r = df['close'].pct_change()
    neg = r.clip(upper=0.0).rolling(60).std()
    pos = r.clip(lower=0.0).rolling(60).std()
    return neg / pos.replace(0, np.nan)

def f_eff_ratio_60(df, s):
    num = (df['close'] - df['close'].shift(60)).abs()
    den = df['close'].pct_change().abs().rolling(60).sum()
    return num / den.replace(0, np.nan)

CAND_DEFS = {
    'kurt_term_20_60':      (f_kurt_term, 'kurt20(ret)-kurt60(ret)'),
    'ovn_gap_mom_20':       (f_ovn_gap_mom, 'mean20(open/close.shift(1)-1)'),
    'intraday_share_20':    (f_intraday_share, 'mean20(intraday ret)-mean20(ovn ret)'),
    'rel_str_xau_60':       (f_rel_str_xau, 'ret60 - ret60(XAU)'),
    'gain_loss_ratio_60':   (f_gain_loss_ratio, 'mean(pos ret)/|mean(neg ret)| 60d'),
    'atr_ratio_20_60':      (f_atr_ratio, 'mean20(TR)/mean60(TR)-1'),
    'ndx_beta_60':          (f_ndx_beta, 'BETA(ret,NDX_ret,60)'),
    'usdjpy_beta_cond_60x20': (f_usdjpy_beta_cond, 'BETA60(ret,USDJPY)*MOM20(USDJPY)'),
    'usdcny_beta_cond_60x20': (f_usdcny_beta_cond, 'BETA60(ret,USDCNY)*MOM20(USDCNY)'),
    'dd_depth_60':          (f_dd_depth, 'close/rolling_max(close,60)-1'),
    'ovn_var_share_20':     (f_ovn_var_share, 'var20(ovn ret)/var20(total ret)'),
    'skew_60d':             (f_skew60, 'skewness60(ret)'),
    'downside_vol_ratio_60':(f_downside_vol_ratio, 'std60(neg ret)/std60(pos ret)'),
    'eff_ratio_60':         (f_eff_ratio_60, '|close-close60|/sum60(|ret|)'),
}

print('\n===== CANDIDATE VALIDATION (h=10) =====', flush=True)
results = {}
for fid, (fn, expr) in CAND_DEFS.items():
    panel = pd.DataFrame({s: fn(prices[s], s) for s in WATCHLIST})
    panel = panel[~panel.index.duplicated(keep='last')].sort_index()
    m = validate(panel)
    rho, rho_id = pairwise_rho_vs_library(panel, lib_panels)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = rho_id
    results[fid] = m
    gate_ic = abs(m['ic']) >= 0.007
    gate_icir = abs(m['icir']) >= 0.084
    gate_rho = rho < 0.5
    print(f'{fid:24s} IC={m["ic"]:+.4f} ICIR={m["icir"]:+.4f} hit={m["ic_hit_ratio"]:.3f} '
          f'n={m["n"]} cov={m["cov"]:.2f} ge8={m["ge8"]:.2f} turn={m["turn"]:.2f} '
          f'rho={rho:.3f}({rho_id})', flush=True)
    print(f'{"":24s}   decay={ {k: round(v,4) for k,v in m["decay"].items()} }', flush=True)
    print(f'{"":24s}   GATE: |IC|>=.007 {gate_ic} |ICIR|>=.084 {gate_icir} rho<.5 {gate_rho} '
          f'-> {"PASS" if (gate_ic and gate_icir and gate_rho) else "FAIL"}', flush=True)

json.dump({fid: {k: v for k, v in m.items() if k != 'decay'} for fid, m in results.items()},
          open('scripts/miner_2_20260730_results_round7.json', 'w'), indent=2, default=str)
print(f'\n[total] {time.time()-t0:.1f}s', flush=True)
