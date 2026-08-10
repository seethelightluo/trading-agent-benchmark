"""miner_1 batch8: screen novel factor candidates on the 15-asset cross-asset universe.

Each candidate is validated individually (IC/ICIR gate on 10d forward returns,
window 2020-01-01..2026-07-15, min 8 valid instruments per date). Full-library
pairwise Spearman correlation is computed against ALL effective factor signal
artifacts in factors/ (not just the 4 reference factors).

Novel ideas in this batch (none persisted before):
  1. rv_bv_ratio_20      - realized vol / bipower variation (jump intensity)
  2. kurt_term_20_60     - kurtosis term structure (kurt20 - kurt60)
  3. gk_vol_ratio_20     - Garman-Klass vol / close-to-close vol (intraday efficiency)
  4. nocturnal_ratio_20  - overnight vol / intraday vol
  5. updown_vol_ratio_20 - upside semi-vol / downside semi-vol
  6. drawdown_depth_60   - distance from 60d high
  7. norm_pos_20_60      - (close - MA20) / (high60 - low60)
  8. streak_20           - consecutive same-sign day streak
  9. lin_tstat_60        - t-stat of linear time trend in close (rolling corr)
  10. hi_lo_time_60      - days since 60d high minus days since 60d low
  11. sharpe_chg_20_60   - Sharpe(20d) - Sharpe(60d)
  12. tail_ratio_20      - 95pct abs ret / median abs ret
"""
import sys, os, json, glob
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, VAL_START, VAL_END, load_prices,
                           canonical_grid, factor_to_panel, validate_factor)

prices = load_prices(days=2000)
grid = canonical_grid(prices)
print(f"prices loaded: {len(prices)} assets; canonical grid {len(grid)} dates "
      f"{grid.min().date()}..{grid.max().date()}")

# ---- load all effective library signal artifacts for full-library rho audit ----
lib_panels = {}
for p in sorted(glob.glob('factors/*_signal.npy')):
    fid = os.path.basename(p).replace('_signal.npy', '')
    arr = np.load(p, allow_pickle=False)
    if arr.shape[0] == len(grid) and arr.shape[1] == len(WATCHLIST):
        lib_panels[fid] = pd.DataFrame(arr, index=grid, columns=WATCHLIST)
    else:
        print(f"  WARN artifact {fid} shape {arr.shape} != grid x {len(WATCHLIST)}")
print(f"library artifacts loaded for rho audit: {len(lib_panels)} -> {sorted(lib_panels)}")


def max_lib_rho(panel):
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


# ---- candidate factor functions (one idea each) ----
def f_rv_bv(df, s):
    r = df['close'].pct_change()
    rv = (r ** 2).rolling(20).sum()
    bv = (r.abs() * r.shift(1).abs()).rolling(20).sum()
    return rv / bv

def f_kurt_term(df, s):
    r = df['close'].pct_change()
    k20 = r.rolling(20).kurt()
    k60 = r.rolling(60).kurt()
    out = k20 - k60
    return out.clip(-20, 20)

def f_gk_ratio(df, s):
    o, h, l, c = df['open'], df['high'], df['low'], df['close']
    lo_c, lo_o, lo_h, lo_l = np.log(c), np.log(o), np.log(h), np.log(l)
    gk = 0.5 * (lo_h - lo_l) ** 2 - (2 * np.log(2) - 1) * (lo_c - lo_o) ** 2
    gk_vol = np.sqrt(gk.clip(lower=0).rolling(20).mean())
    ctc = c.pct_change().rolling(20).std()
    return (gk_vol / ctc).clip(0, 5)

def f_nocturnal(df, s):
    on = df['open'] / df['close'].shift(1) - 1.0
    intra = df['close'] / df['open'] - 1.0
    on_vol = on.rolling(20).std()
    intra_vol = intra.rolling(20).std()
    return (on_vol / intra_vol).clip(0, 5)

def f_updown_vol(df, s):
    r = df['close'].pct_change()
    up = r.where(r > 0)
    dn = r.where(r < 0)
    return (up.rolling(20).std() / dn.rolling(20).std()).clip(0, 5)

def f_dd_depth(df, s):
    hh = df['close'].rolling(60).max()
    return (df['close'] / hh - 1.0).clip(-1, 0)

def f_norm_pos(df, s):
    c = df['close']
    ma = c.rolling(20).mean()
    hi = c.rolling(60).max()
    lo = c.rolling(60).min()
    return ((c - ma) / (hi - lo)).clip(-3, 3)

def f_streak(df, s):
    sign = np.sign(df['close'].pct_change()).fillna(0)
    grp = (sign != sign.shift()).cumsum()
    streak = (sign.groupby(grp).cumcount() + 1) * sign
    return streak

def f_lin_tstat(df, s):
    n = 60
    y = df['close']
    x = pd.Series(np.arange(len(y)), index=y.index)
    c = y.rolling(n).corr(x)
    t = c * np.sqrt(n - 2) / np.sqrt((1 - c ** 2).clip(lower=1e-12))
    return t.clip(-20, 20)

def f_hilo_time(df, s):
    n = 60
    since_high = df['close'].rolling(n).apply(lambda w: len(w) - 1 - int(np.argmax(w)), raw=True)
    since_low = df['close'].rolling(n).apply(lambda w: len(w) - 1 - int(np.argmin(w)), raw=True)
    return since_high - since_low

def f_sharpe_chg(df, s):
    r = df['close'].pct_change()
    sh20 = r.rolling(20).mean() / r.rolling(20).std()
    sh60 = r.rolling(60).mean() / r.rolling(60).std()
    return (sh20 - sh60).clip(-5, 5)

def f_tail_ratio(df, s):
    r = df['close'].pct_change().abs()
    p95 = r.rolling(20).quantile(0.95)
    med = r.rolling(20).median()
    return (p95 / med).clip(0, 10)

CANDIDATES = [
    ('rv_bv_ratio_20', f_rv_bv, ['jump', 'volatility']),
    ('kurt_term_20_60', f_kurt_term, ['kurtosis', 'tail-risk']),
    ('gk_vol_ratio_20', f_gk_ratio, ['volatility', 'intraday']),
    ('nocturnal_ratio_20', f_nocturnal, ['volatility', 'overnight']),
    ('updown_vol_ratio_20', f_updown_vol, ['volatility', 'asymmetry']),
    ('drawdown_depth_60', f_dd_depth, ['drawdown', 'trend']),
    ('norm_pos_20_60', f_norm_pos, ['mean-reversion', 'position']),
    ('streak_20', f_streak, ['momentum', 'microstructure']),
    ('lin_tstat_60', f_lin_tstat, ['trend', 'momentum']),
    ('hi_lo_time_60', f_hilo_time, ['cycle', 'position']),
    ('sharpe_chg_20_60', f_sharpe_chg, ['momentum', 'risk-adjusted']),
    ('tail_ratio_20', f_tail_ratio, ['tail', 'volatility']),
]

print("\n" + "=" * 100)
for fid, fn, tags in CANDIDATES:
    panel = factor_to_panel(fn, prices)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f"{fid}: insufficient data -> None")
        continue
    rho, rho_id = max_lib_rho(panel)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = rho_id
    ic, icir = m['ic'], m['icir']
    ok_ic = abs(ic) >= 0.007
    ok_icir = abs(icir) >= 0.084
    ok_rho = rho < 0.5
    ok = ok_ic and ok_icir and ok_rho
    print(f"[{'PASS' if ok else 'FAIL'}] {fid:20s} IC={ic:+.4f} ICIR={icir:+.4f} "
          f"hit={m['ic_hit_ratio']:.3f} cov={m['coverage_asset_days']:.3f} "
          f"ge8={m['coverage_dates_ge8']:.3f} turn={m['turnover_10d_rank']:.2f} "
          f"rho={rho:.3f}({rho_id}) n={m['n_ic_dates']}")
    print(f"     decay: " + " ".join(f"{h}:{v:+.4f}" for h, v in m['decay_ic_by_horizon'].items()))
print("=" * 100)
print("done")
