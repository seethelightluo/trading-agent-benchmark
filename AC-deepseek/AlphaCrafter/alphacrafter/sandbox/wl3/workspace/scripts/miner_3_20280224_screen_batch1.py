"""miner_3 2028-02-24 batch-1 novel factor screen.

Candidate ideas (cross-asset, interpretable):
1. vwap_mom_20        - volume-weighted 20d momentum (volume-confirmed trend)
2. obv_slope_20       - On-Balance-Volume 20d slope normalized by volume
3. universe_beta_60   - rolling beta vs equal-weight 15-asset universe return
4. downside_vol_share_60 - share of vol coming from down days (60d)
5. rolling_sharpe_60  - 60d mean/std return (risk-adjusted drift)
6. kurtosis_60        - 60d excess kurtosis of daily returns
7. crypto_basket_beta_60 - rolling beta vs (BTC+ETH)/2 return
8. co_skew_60         - rolling regression of squared residuals on market ret
9. us10y_beta_60      - rolling beta vs US10Y daily return
10. dd_60             - current drawdown from 60d high
11. skew_60_total     - 60d skewness of daily total returns
12. updown_vol_ratio_60 - std(pos days)/std(neg days) over 60d

Admission gates: |IC10| >= 0.007, |ICIR10| >= 0.084 on 2020-01-01..2026-07-15.
Also reports recent-window IC (2026-07-16..2028-02-23) for drift/timeliness.
Max abs library correlation is computed against ALL root signal artifacts
(48 npy) - conservative audit.
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, 'scripts')
from factor_common import (load_prices, load_index, canonical_grid, WATCHLIST,
                           VAL_START, VAL_END, factor_to_panel, forward_returns,
                           rank_ic_series, signal_matrix)

prices = load_prices(days=2600)
print(f"assets loaded: {len(prices)}; max date: {max(d.index.max() for d in prices.values())}")

# ---------- helpers ----------
def rolling_beta(df, mkt, window):
    r = df['close'].pct_change()
    mm = mkt.reindex(r.index).ffill()
    z = pd.concat([r.rename('r'), mm.rename('m')], axis=1).dropna()
    cov = z['r'].rolling(window).cov(z['m'])
    var = z['m'].rolling(window).var()
    return (cov / var).reindex(df.index)

def rolling_skew(s, window):
    return s.rolling(window).skew()

def rolling_kurt(s, window):
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std()
    return (((s - mu) ** 4).rolling(window).mean() / (sd ** 4) - 3.0)

# market returns for universe-level factors
r_all = pd.DataFrame({s: d['close'].pct_change() for s, d in prices.items()}).sort_index()
r_ew = r_all.mean(axis=1, min_count=8)
r_btc = r_all['BTC']; r_eth = r_all['ETH']
r_crypto = (r_btc + r_eth) / 2.0
us10y_df = prices['US10Y']
r_us10y = us10y_df['close'].pct_change()

candidates = {}

def f_vwap_mom(df, s):
    r = df['close'].pct_change(); v = df['volume'].replace(0, np.nan)
    return (r * v).rolling(20).sum() / v.rolling(20).sum()
candidates['vwap_mom_20'] = f_vwap_mom

def f_obv_slope(df, s):
    r = df['close'].pct_change()
    v = df['volume'].replace(0, np.nan)
    obv = (np.sign(r) * v).cumsum()
    return (obv - obv.shift(20)) / (v.rolling(20).mean() * 20.0)
candidates['obv_slope_20'] = f_obv_slope

def f_universe_beta(df, s):
    return rolling_beta(df, r_ew, 60)
candidates['universe_beta_60'] = f_universe_beta

def f_downside_vol_share(df, s):
    r = df['close'].pct_change()
    neg = r.clip(upper=0.0)
    return neg.rolling(60).std() / r.rolling(60).std()
candidates['downside_vol_share_60'] = f_downside_vol_share

def f_roll_sharpe(df, s):
    r = df['close'].pct_change()
    return r.rolling(60).mean() / r.rolling(60).std()
candidates['rolling_sharpe_60'] = f_roll_sharpe

def f_kurt_60(df, s):
    return rolling_kurt(df['close'].pct_change(), 60)
candidates['kurtosis_60'] = f_kurt_60

def f_crypto_beta(df, s):
    return rolling_beta(df, r_crypto, 60)
candidates['crypto_basket_beta_60'] = f_crypto_beta

def f_coskew(df, s):
    r = df['close'].pct_change()
    m = r_ew.reindex(r.index).ffill()
    z = pd.concat([r.rename('r'), m.rename('m')], axis=1).dropna()
    mu = z['r'].rolling(60).mean()
    sq = (z['r'] - mu) ** 2
    cov = sq.rolling(60).cov(z['m']); var = z['m'].rolling(60).var()
    return (cov / var).reindex(df.index)
candidates['co_skew_60'] = f_coskew

def f_us10y_beta(df, s):
    return rolling_beta(df, r_us10y, 60)
candidates['us10y_beta_60'] = f_us10y_beta

def f_dd_60(df, s):
    return df['close'] / df['close'].rolling(60).max() - 1.0
candidates['dd_60'] = f_dd_60

def f_skew_total(df, s):
    return rolling_skew(df['close'].pct_change(), 60)
candidates['skew_60_total'] = f_skew_total

def f_updown_vol(df, s):
    r = df['close'].pct_change()
    pos = r.clip(lower=0.0); neg = r.clip(upper=0.0)
    return pos.rolling(60).std() / neg.rolling(60).std().abs()
candidates['updown_vol_ratio_60'] = f_updown_vol

# ---------- library artifacts for correlation audit ----------
grid = canonical_grid(prices)
lib_artifacts = {}
for p in sorted(Path('factors').glob('*_signal.npy')):
    arr = np.load(p, allow_pickle=False)
    if arr.shape[0] == len(grid) and arr.shape[1] == 15:
        lib_artifacts[p.name.replace('_signal.npy', '')] = arr
print(f"library artifacts for corr audit: {len(lib_artifacts)}")

def max_lib_corr(mat):
    best, best_id = 0.0, None
    for fid, la in lib_artifacts.items():
        corrs = []
        for i in range(len(grid)):
            x = mat[i]; y = la[i]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                c = pd.Series(x[m]).rank().corr(pd.Series(y[m]).rank())
                if np.isfinite(c):
                    corrs.append(c)
        if corrs:
            r = float(np.mean(corrs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id

# ---------- validation ----------
fwd = {h: forward_returns(prices, h) for h in (1, 2, 3, 5, 10, 20)}
results = {}
for fid, fn in candidates.items():
    panel = factor_to_panel(fn, prices)
    if panel.empty:
        print(f"{fid}: EMPTY panel"); continue
    ic10 = rank_ic_series(panel, fwd[10])
    ic10_w = ic10[(ic10.index >= VAL_START) & (ic10.index <= VAL_END)]
    if len(ic10_w) < 100:
        print(f"{fid}: insufficient warm-up IC dates {len(ic10_w)}"); continue
    ic = float(ic10_w.mean()); icir = ic / float(ic10_w.std(ddof=1)) if ic10_w.std(ddof=1) > 0 else 0.0
    hit = float((ic10_w > 0).mean()) if ic >= 0 else float((ic10_w < 0).mean())
    # coverage in warm-up
    fac = panel[(panel.index >= VAL_START) & (panel.index <= VAL_END)]
    cov = float(fac.notna().sum().sum()) / (fac.shape[0] * fac.shape[1]) if fac.shape[0] else 0.0
    ge8 = float((fac.notna().sum(axis=1) >= 8).mean())
    turn = float(fac.rank(axis=1).diff(10).abs().mean().mean()) if len(fac) > 10 else float('nan')
    decay = {str(h): float(rank_ic_series(panel, fwd[h]).loc[lambda s: (s.index >= VAL_START) & (s.index <= VAL_END)].mean()) for h in (1, 2, 3, 5, 10, 20)}
    # recent-window drift check (2026-07-16 .. max visible)
    rstart = VAL_END + pd.Timedelta(days=1)
    ic_recent = rank_ic_series(panel, fwd[10])
    ic_recent = ic_recent[ic_recent.index >= rstart]
    ic_rc = float(ic_recent.mean()) if len(ic_recent) >= 30 else float('nan')
    icir_rc = ic_rc / float(ic_recent.std(ddof=1)) if len(ic_recent) >= 30 and ic_recent.std(ddof=1) > 0 else float('nan')
    # library corr
    mat = signal_matrix(panel, grid)
    rho, fid_rho = max_lib_corr(mat)
    ok = abs(ic) >= 0.007 and abs(icir) >= 0.084
    results[fid] = dict(ic=ic, icir=icir, hit=hit, cov=cov, ge8=ge8, turn=turn,
                        decay=decay, rho=rho, rho_id=fid_rho,
                        ic_recent=ic_rc, icir_recent=icir_rc, n_recent=len(ic_recent),
                        n_warm=len(ic10_w))
    print(f"\n{fid}: warm IC={ic:.4f} ICIR={icir:.4f} hit={hit:.3f} cov={cov:.3f} ge8={ge8:.3f} turn={turn:.2f}")
    print(f"   decay: " + " ".join(f"{h}:{decay[str(h)]:.4f}" for h in (1,2,3,5,10,20)))
    print(f"   recent(2026-07-16+): IC={ic_rc:.4f} ICIR={icir_rc:.4f} n={len(ic_recent)}")
    print(f"   max|lib rho|={rho:.4f} vs {fid_rho}")
    print(f"   ADMISSION: {'PASS' if ok else 'FAIL'}")

print("\n=== SUMMARY ===")
for fid, r in results.items():
    print(f"{fid:24s} IC={r['ic']:.4f} ICIR={r['icir']:.4f} rho={r['rho']:.3f} recentIC={r['ic_recent']:.4f} PASS={'Y' if abs(r['ic'])>=0.007 and abs(r['icir'])>=0.084 else 'N'}")
