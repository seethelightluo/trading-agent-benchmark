"""miner_1 clean vectorized factor validation (2026-07-30 cycle).

Panel: 15 tradables on the common macro weekday calendar, truncated at
visible_through (2026-07-29). No future data.
Metrics: daily cross-sectional Spearman rank IC at horizons 1..20,
ICIR = mean(abs-sign-consistent IC)/std, hit ratio, coverage, turnover,
sub-period robustness, and pairwise correlation among candidates.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MACRO = ['DXY', 'VIX', 'USDCNY', 'USDJPY', 'EURUSD']
IC_TH, ICIR_TH = 0.0070, 0.0840
MIN_ASSETS = 8

DATE_JSON = Path('../persistent/date.json')
STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')


def visible_through():
    d = json.load(open(DATE_JSON))
    return pd.to_datetime(d.get('visible_through', d.get('current_date')))


def load_macro_panel():
    vth = visible_through()
    frames = {}
    for s in MACRO:
        df = pd.read_csv(INDEX_DIR / f'{s}.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] <= vth].sort_values('date')
        frames[s] = df.set_index('date')['close'].astype(float)
    return pd.DataFrame(frames).sort_index()


def load_panel():
    vth = visible_through()
    macro = load_macro_panel()
    cal = macro.index
    closes, vols = {}, {}
    for s in WATCH:
        df = pd.read_csv(STOCK_DIR / f'{s}.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] <= vth].sort_values('date')
        srs = df.set_index('date')['close'].astype(float).reindex(cal)
        closes[s] = srs
        vols[s] = (df.set_index('date')['volume'].astype(float).reindex(cal)
                   if 'volume' in df else pd.Series(np.nan, index=cal))
    return pd.DataFrame(closes).sort_index(), pd.DataFrame(vols).sort_index()


def spearman_ic_series(factor, fwd):
    """Vectorized per-date Spearman IC between factor and forward returns."""
    fa = factor.reindex(fwd.index)
    A = fa.rank(axis=1, pct=True)
    B = fwd.rank(axis=1, pct=True)
    m = A.notna().values & B.notna().values
    n = m.sum(axis=1)
    ok = n >= MIN_ASSETS
    A = np.nan_to_num(A.values, nan=0.0)
    B = np.nan_to_num(B.values, nan=0.0)
    Ac = A - np.where(m, A, 0).sum(axis=1, keepdims=True) / np.maximum(n, 1)[:, None]
    Bc = B - np.where(m, B, 0).sum(axis=1, keepdims=True) / np.maximum(n, 1)[:, None]
    Ac = np.where(m, Ac, 0.0)
    Bc = np.where(m, Bc, 0.0)
    cov = (Ac * Bc).sum(axis=1) / np.maximum(n - 1, 1)
    va = (Ac * Ac).sum(axis=1)
    vb = (Bc * Bc).sum(axis=1)
    denom = np.sqrt(va * vb)
    ic = np.full(len(n), np.nan)
    okd = ok & (denom > 1e-12)
    ic[okd] = cov[okd] / denom[okd]
    return pd.Series(ic, index=fwd.index)


def fast_turnover(factor):
    """Mean abs change of cross-sectional rank (pct) per 10 trading days."""
    r = factor.rank(axis=1, pct=True)
    d = r.diff(10).abs().mean(axis=1)
    return float(d.mean())


def coverage_stats(factor):
    cells = factor.notna().sum().sum()
    total = factor.shape[0] * factor.shape[1]
    dates_ge8 = float((factor.notna().sum(axis=1) >= MIN_ASSETS).mean())
    return cells / total, dates_ge8


def ic_analysis(factor, fwd_map, label):
    ic_series = {}
    for h, fwd in fwd_map.items():
        ic_series[h] = spearman_ic_series(factor, fwd)
    ic10 = ic_series[10].dropna()
    if len(ic10) == 0:
        return None
    ic = float(ic10.mean())
    std = float(ic10.std(ddof=1))
    icir = ic / std if std > 0 else np.nan
    hit = float((ic10 > 0).mean())
    cov_ad, cov_d = coverage_stats(factor)
    turn = fast_turnover(factor)
    decay = {h: (round(float(s.mean()), 4) if s.notna().any() else None)
             for h, s in ic_series.items()}
    sub = {}
    for lo, hi, tag in [('2020-01-01', '2022-12-31', '2020-2022'),
                        ('2023-01-01', '2026-07-29', '2023-2026')]:
        s = ic10[(ic10.index >= lo) & (ic10.index <= hi)]
        sub[tag] = (round(float(s.mean()), 4), int(len(s))) if len(s) else None
    return {'label': label, 'horizon': 10, 'ic': ic, 'icir': icir, 'hit': hit,
            'n_ic_dates': int(len(ic10)), 'cov_asset_days': cov_ad,
            'cov_dates_ge8': cov_d, 'turnover_10d': turn, 'decay': decay,
            'subperiod': sub}


def pairwise_corr(factors: dict):
    names = list(factors.keys())
    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            fa = factors[a].rank(axis=1, pct=True)
            fb = factors[b].rank(axis=1, pct=True)
            idx = fa.index.intersection(fb.index)
            A = np.nan_to_num(fa.loc[idx].values, nan=0.0)
            B = np.nan_to_num(fb.loc[idx].values, nan=0.0)
            m = np.isfinite(fa.loc[idx].values) & np.isfinite(fb.loc[idx].values)
            n = m.sum(axis=1)
            ok = n >= MIN_ASSETS
            Ac = A - np.where(m, A, 0).sum(axis=1, keepdims=True) / np.maximum(n, 1)[:, None]
            Bc = B - np.where(m, B, 0).sum(axis=1, keepdims=True) / np.maximum(n, 1)[:, None]
            Ac = np.where(m, Ac, 0.0); Bc = np.where(m, Bc, 0.0)
            cov = (Ac * Bc).sum(axis=1) / np.maximum(n - 1, 1)
            den = np.sqrt((Ac * Ac).sum(axis=1) * (Bc * Bc).sum(axis=1))
            rho = cov[ok & (den > 1e-12)] / den[ok & (den > 1e-12)]
            v = float(np.mean(rho)) if len(rho) else np.nan
            mat.loc[a, b] = mat.loc[b, a] = v
    np.fill_diagonal(mat.values, 1.0)
    return mat


def main():
    panel, vol = load_panel()
    ret = panel.pct_change()
    macro = load_macro_panel()
    print(f'panel: {panel.shape} dates={len(panel)} assets={panel.shape[1]} '
          f'window {panel.index[0].date()}..{panel.index[-1].date()}')
    fwd = {h: panel.shift(-h) / panel - 1.0 for h in (1, 2, 3, 5, 10, 20)}
    fwd10 = fwd[10]

    def roll_std(x, w, mp=None):
        return x.rolling(w, min_periods=mp or max(10, w // 2)).std()

    def roll_mean(x, w, mp=None):
        return x.rolling(w, min_periods=mp or max(10, w // 2)).mean()

    def beta_of(a, m, w):
        return a.rolling(w, min_periods=max(20, w // 2)).cov(m) / \
               m.rolling(w, min_periods=max(20, w // 2)).var()

    C = {}
    # ---- momentum family (5d skip) ----
    for lb in (10, 20, 60, 120, 180, 250):
        C[f'mom_{lb}d_skip5'] = panel.shift(5) / panel.shift(lb + 5) - 1.0
    C['mom_20d'] = panel / panel.shift(20) - 1.0
    C['mom_60d'] = panel / panel.shift(60) - 1.0
    C['risk_adj_mom_20d_skip5'] = (panel.shift(5) / panel.shift(25) - 1.0) / roll_std(ret, 20)
    # ---- volatility / risk family ----
    C['vol_20d'] = roll_std(ret, 20)
    C['inv_vol_60d'] = -roll_std(ret, 60)
    C['vol_of_vol20x60'] = roll_std(roll_std(ret, 20), 60)
    C['vol_ratio_10x60'] = roll_std(ret, 10) / roll_std(ret, 60)
    C['downside_ratio_20x60'] = (ret.clip(upper=0).rolling(20, min_periods=10).std() /
                                 ret.clip(upper=0).rolling(60, min_periods=30).std())
    C['skew_60d'] = ret.rolling(60, min_periods=30).skew()
    C['kurt_60d'] = ret.rolling(60, min_periods=30).kurt()
    C['mdd_60d'] = panel / panel.rolling(60, min_periods=30).max() - 1.0
    C['dist_52w_low'] = panel / panel.rolling(250, min_periods=125).min() - 1.0
    C['eff_ratio_20d'] = ret.rolling(20, min_periods=10).mean() / roll_std(ret, 20)
    C['range_20d'] = (panel.rolling(20, min_periods=10).max() -
                      panel.rolling(20, min_periods=10).min()) / panel
    # ---- macro beta family ----
    dxy_r, vix_r = macro['DXY'].pct_change(), macro['VIX'].pct_change()
    jpy_r, cny_r, eur_r = (macro['USDJPY'].pct_change(), macro['USDCNY'].pct_change(),
                           macro['EURUSD'].pct_change())
    us10y_r, cn10y_r = panel['US10Y'].pct_change(), panel['CN10Y'].pct_change()
    for name, m in [('dxy', dxy_r), ('vix', vix_r), ('jpy', jpy_r),
                    ('cny', cny_r), ('eur', eur_r), ('us10y', us10y_r),
                    ('cn10y', cn10y_r)]:
        C[f'beta_{name}_60d'] = beta_of(ret, m, 60)
    C['vix_beta_cond_60x20'] = -beta_of(ret, vix_r, 60) * (macro['VIX'] / macro['VIX'].shift(20) - 1.0)
    C['dxy_beta_cond_60x20'] = -beta_of(ret, dxy_r, 60) * (macro['DXY'] / macro['DXY'].shift(20) - 1.0)
    C['jpy_beta_cond_60x20'] = -beta_of(ret, jpy_r, 60) * (macro['USDJPY'] / macro['USDJPY'].shift(20) - 1.0)
    # ---- cross-asset relative momentum ----
    C['wti_copper_rel_20d'] = panel['WTI'].pct_change(20) - panel['COPPER'].pct_change(20)
    C['crypto_equity_rel_20d'] = (panel['BTC'].pct_change(20) + panel['ETH'].pct_change(20)) / 2 - \
        (panel['NDX'].pct_change(20) + panel['SPX'].pct_change(20)) / 2
    C['gold_bond_rel_20d'] = panel['XAU'].pct_change(20) - panel['US10Y'].pct_change(20)
    C['eq_bond_rel_20d'] = (panel['SPX'].pct_change(20) + panel['NDX'].pct_change(20)) / 2 - \
        panel['US10Y'].pct_change(20)
    # ---- liquidity ----
    if vol.notna().sum().sum() > 0:
        C['amihud_20d'] = (ret.abs() / panel.replace(0, np.nan)).rolling(20, min_periods=10).mean() \
            * (vol / vol.rolling(20, min_periods=10).mean()).clip(lower=0.2, upper=5)

    results = {}
    for name, f in C.items():
        res = ic_analysis(f, fwd, name)
        if res is None:
            print(f'{name:<26} NO_DATA')
            continue
        results[name] = res
        gate = abs(res['ic']) >= IC_TH and abs(res['icir']) >= ICIR_TH
        print(f"{name:<26} ic={res['ic']:+.4f} icir={res['icir']:.4f} hit={res['hit']:.3f} "
              f"n={res['n_ic_dates']:>5d} cov={res['cov_asset_days']:.3f}/{res['cov_dates_ge8']:.2f} "
              f"turn={res['turnover_10d']:.2f} {'PASS' if gate else ''}")
        print(f"    decay={res['decay']} sub={res['subperiod']}")

    print('\n=== top by |ic|*|icir| (abs) ===')
    scored = sorted(results.items(), key=lambda kv: abs(kv[1]['ic']) * abs(kv[1]['icir']), reverse=True)
    for name, r in scored[:15]:
        print(f"  {name:<26} ic={r['ic']:+.4f} icir={r['icir']:.4f} |ic*icir|={abs(r['ic']*r['icir']):.5f} "
              f"n={r['n_ic_dates']} cov={r['cov_asset_days']:.3f}")

    # pairwise correlation among PASSING candidates
    passed = {n: C[n] for n, r in results.items()
              if abs(r['ic']) >= IC_TH and abs(r['icir']) >= ICIR_TH and r['n_ic_dates'] >= 200}
    print(f'\n=== pairwise |rho| among {len(passed)} passing candidates ===')
    if len(passed) >= 2:
        mat = pairwise_corr(passed)
        print(mat.round(3).to_string())

    # save candidate signals for persistence step
    import pickle
    with open('scripts/miner_1_20260730_candidates.pkl', 'wb') as fh:
        pickle.dump({n: C[n] for n in passed}, fh)
    print('\nsaved passing candidates to scripts/miner_1_20260730_candidates.pkl')


if __name__ == '__main__':
    main()
