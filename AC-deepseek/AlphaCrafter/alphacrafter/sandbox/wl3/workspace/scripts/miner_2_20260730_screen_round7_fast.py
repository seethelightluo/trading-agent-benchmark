"""miner_2 2026-07-30 round-7 factor screen (vectorized fast path).

Candidates:
  round6 reruns: kurt_term_20_60, ovn_gap_mom_20, intraday_share_20,
    rel_str_xau_60, gain_loss_ratio_60, atr_ratio_20_60, ndx_beta_60
  new round7: usdjpy_beta_cond_60x20, usdcny_beta_cond_60x20, dd_depth_60,
    ovn_var_share_20, skew_60d, downside_vol_ratio_60, eff_ratio_60

Gate: |IC|>=0.007, |ICIR|>=0.084 (h=10, 2020-01-01..2026-07-15),
rho_vs_library < 0.5. Library panels loaded from persisted signal artifacts.
"""
import sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, load_index, canonical_grid,
                           WATCHLIST, VAL_START, VAL_END)

t0 = time.time()
prices = load_prices(days=2600)
dxy = load_index('DXY', prices=prices)
eurusd = load_index('EURUSD', prices=prices)
usdjpy = load_index('USDJPY', prices=prices)
usdcny = load_index('USDCNY', prices=prices)
vix = load_index('VIX', prices=prices)
grid = canonical_grid(prices)
T = len(grid); N = len(WATCHLIST)
print(f'[load] {len(prices)} assets; grid {T} dates '
      f'{grid.min().date()}..{grid.max().date()}', flush=True)

close = np.full((T, N), np.nan)
for i, s in enumerate(WATCHLIST):
    close[:, i] = prices[s]['close'].reindex(grid).values

# forward return matrices
fwd = {}
for h in (1, 2, 3, 5, 10, 20):
    y = np.full_like(close, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        y[:-h] = close[h:] / close[:-h] - 1.0
    fwd[h] = y


def row_rank(X, m):
    T_, N_ = X.shape
    R = np.full_like(X, np.nan)
    for t in range(T_):
        idx = np.where(m[t])[0]
        if len(idx) >= 3:
            o = np.argsort(X[t, idx], kind='mergesort')
            r = np.empty(len(idx))
            r[o] = np.arange(1.0, len(idx) + 1.0)
            R[t, idx] = r
    return R


def ic_series(F, Y, min_valid=8):
    m = np.isfinite(F) & np.isfinite(Y)
    RF = row_rank(F, m)
    RY = row_rank(Y, m)
    out = np.full(T, np.nan)
    for t in range(T):
        mm = m[t]
        if mm.sum() >= min_valid:
            a = RF[t, mm]; b = RY[t, mm]
            a = a - a.mean(); b = b - b.mean()
            den = np.sqrt((a * a).sum() * (b * b).sum())
            if den > 0:
                out[t] = float((a * b).sum() / den)
    return out


def validate(F):
    ic_s = {h: ic_series(F, fwd[h]) for h in fwd}
    ic10 = ic_s[10]
    full = ic10[(grid >= VAL_START) & (grid <= VAL_END)]
    m = float(full.mean()); sd = float(full.std(ddof=1))
    icir = m / sd if sd > 0 else 0.0
    hit = float((full > 0).mean()) if m >= 0 else float((full < 0).mean())
    fac = F[(grid >= VAL_START) & (grid <= VAL_END)]
    cov = float(np.isfinite(fac).sum()) / fac.size
    ge8 = float((np.isfinite(fac).sum(axis=1) >= 8).mean())
    rk = np.full_like(fac, np.nan)
    for t in range(fac.shape[0]):
        mm = np.isfinite(fac[t])
        if mm.sum() >= 3:
            o = np.argsort(fac[t, mm], kind='mergesort')
            r = np.empty(mm.sum()); r[o] = np.arange(1.0, mm.sum() + 1.0)
            rk[t, mm] = r
    diff = np.abs(np.diff(rk, n=10, axis=0)) if len(rk) > 10 else np.full((1, N), np.nan)
    turn = float(np.nanmean(diff))
    decay = {str(h): float(np.nanmean(ic_s[h])) for h in fwd}
    return dict(ic=m, icir=icir, ic_hit_ratio=hit, n=int(len(full)), cov=cov, ge8=ge8,
                turn=turn, decay=decay)


def pairwise_rho_vs_library(F, lib):
    best, best_id = 0.0, None
    for fid, L in lib.items():
        m = np.isfinite(F) & np.isfinite(L)
        RF = row_rank(F, m); RL = row_rank(L, m)
        corrs = []
        for t in range(T):
            mm = m[t]
            if mm.sum() >= 8:
                a = RF[t, mm]; b = RL[t, mm]
                a = a - a.mean(); b = b - b.mean()
                den = np.sqrt((a * a).sum() * (b * b).sum())
                if den > 0:
                    corrs.append((a * b).sum() / den)
        if corrs:
            r = float(np.mean(corrs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id


# ---------------- library panels from persisted artifacts ----------------
ACTIVE = ['down_beta_60', 'spx_beta_60', 'vol_adj_mom_20_60', 'dxy_beta_cond_60x20',
          'hs300_beta_60', 'vol_of_vol20x60', 'dd_duration_120_resid',
          'vix_beta_cond_60x20', 'skew_term_20_60', 'hilo_pos_60', 'max_ret_20d',
          'eurusd_beta_cond_60x20']
lib = {}
for fid in ACTIVE:
    p = Path('factors') / f'{fid}_signal.npy'
    arr = np.load(p, allow_pickle=False)
    if arr.shape[0] == T and arr.shape[1] == N:
        lib[fid] = arr.astype(float)
    else:
        lib[fid] = None
lib = {k: v for k, v in lib.items() if v is not None}
print(f'[lib] loaded {len(lib)} library panels from artifacts ({time.time()-t0:.1f}s)', flush=True)

# ---------------- candidates ----------------
def ret(df):
    return df['close'].pct_change().values

def f_kurt_term(df, s):
    r = df['close'].pct_change()
    return (r.rolling(20).kurt() - r.rolling(60).kurt()).values

def f_ovn_gap_mom(df, s):
    ovn = df['open'] / df['close'].shift(1) - 1.0
    return ovn.rolling(20).mean().values

def f_intraday_share(df, s):
    ovn = df['open'] / df['close'].shift(1) - 1.0
    intr = df['close'] / df['open'] - 1.0
    return (intr.rolling(20).mean() - ovn.rolling(20).mean()).values

def f_rel_str_xau(df, s):
    xau = prices['XAU']['close'].reindex(df.index)
    return ((df['close'] / df['close'].shift(60) - 1.0) - (xau / xau.shift(60) - 1.0)).values

def f_gain_loss_ratio(df, s):
    r = df['close'].pct_change()
    pos = r.clip(lower=0.0).rolling(60).mean()
    neg = (-r.clip(upper=0.0)).rolling(60).mean()
    return (pos / neg.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).values

def f_atr_ratio(df, s):
    pc = df['close'].shift(1)
    tr = pd.concat([(df['high'] - df['low']), (df['high'] - pc).abs(),
                    (df['low'] - pc).abs()], axis=1).max(axis=1)
    return (tr.rolling(20).mean() / tr.rolling(60).mean() - 1.0).values

def _beta_anchor(anchor_close, df, s, w=60):
    a = anchor_close.reindex(df.index)
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), a.rename('a')], axis=1).dropna()
    b = z['r'].rolling(w).cov(z['a']) / z['a'].rolling(w).var()
    return b.reindex(df.index).values

def _beta_cond(anchor_close, df, s, sign=1.0):
    a = anchor_close.reindex(df.index)
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), a.rename('a')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['a']) / z['a'].rolling(60).var()
    mom = (anchor_close / anchor_close.shift(20) - 1.0).reindex(df.index)
    return (sign * b * mom).reindex(df.index).values

def f_ndx_beta(df, s):
    return _beta_anchor(prices['NDX']['close'], df, s)

def f_usdjpy_beta_cond(df, s):
    return _beta_cond(usdjpy['close'], df, s)

def f_usdcny_beta_cond(df, s):
    return _beta_cond(usdcny['close'], df, s)

def f_dd_depth(df, s):
    return (df['close'] / df['close'].rolling(60).max() - 1.0).values

def f_ovn_var_share(df, s):
    ovn = df['open'] / df['close'].shift(1) - 1.0
    tot = df['close'].pct_change()
    return (ovn.rolling(20).var() / tot.rolling(20).var().replace(0, np.nan)).values

def f_skew60(df, s):
    return df['close'].pct_change().rolling(60).skew().values

def f_downside_vol_ratio(df, s):
    r = df['close'].pct_change()
    neg = r.clip(upper=0.0).rolling(60).std()
    pos = r.clip(lower=0.0).rolling(60).std()
    return (neg / pos.replace(0, np.nan)).values

def f_eff_ratio_60(df, s):
    num = (df['close'] - df['close'].shift(60)).abs()
    den = df['close'].pct_change().abs().rolling(60).sum()
    return (num / den.replace(0, np.nan)).values

CAND_DEFS = {
    'kurt_term_20_60':       f_kurt_term,
    'ovn_gap_mom_20':        f_ovn_gap_mom,
    'intraday_share_20':     f_intraday_share,
    'rel_str_xau_60':        f_rel_str_xau,
    'gain_loss_ratio_60':    f_gain_loss_ratio,
    'atr_ratio_20_60':       f_atr_ratio,
    'ndx_beta_60':           f_ndx_beta,
    'usdjpy_beta_cond_60x20': f_usdjpy_beta_cond,
    'usdcny_beta_cond_60x20': f_usdcny_beta_cond,
    'dd_depth_60':           f_dd_depth,
    'ovn_var_share_20':      f_ovn_var_share,
    'skew_60d':              f_skew60,
    'downside_vol_ratio_60': f_downside_vol_ratio,
    'eff_ratio_60':          f_eff_ratio_60,
}

def to_panel(fn, s):
    vals = fn(prices[s], s)
    return np.asarray(vals, dtype=float)

print('\n===== CANDIDATE VALIDATION (h=10) =====', flush=True)
results = {}
for fid, fn in CAND_DEFS.items():
    F = np.column_stack([to_panel(fn, s) for s in WATCHLIST])
    m = validate(F)
    rho, rho_id = pairwise_rho_vs_library(F, lib)
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
