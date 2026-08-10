"""Shared validation machinery for factor mining (miner_1, 2026-07-30).

Loads the 15-asset tradable universe + 5 macro observation signals on a common
trading calendar (macro index), computes cross-sectional rank IC / ICIR / hit
ratio / coverage / turnover / decay, and max-abs correlation vs the existing
factor library. Data is truncated to the simulation-visible window
(current_date / visible_through in date.json).
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MACRO = ['DXY', 'VIX', 'USDCNY', 'USDJPY', 'EURUSD']

DATE_JSON = Path('../persistent/date.json')
STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')

IC_TH, ICIR_TH = 0.0070, 0.0840
MIN_ASSETS_PER_DATE = 8


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


def load_close_panel():
    """Close panel on the common trading calendar (macro weekday calendar)."""
    vth = visible_through()
    macro = load_macro_panel()
    cal = macro.index
    closes = {}
    for s in WATCH:
        df = pd.read_csv(STOCK_DIR / f'{s}.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] <= vth].sort_values('date')
        srs = df.set_index('date')['close'].astype(float)
        closes[s] = srs.reindex(cal)
    return pd.DataFrame(closes).sort_index()


def forward_returns(panel, horizons=(1, 2, 3, 5, 10, 20)):
    """Forward close-to-close returns per asset (fractional)."""
    fwd = {}
    for h in horizons:
        fwd[h] = panel.shift(-h) / panel - 1.0
    return fwd


def rank_ic_series(factor_df, fwd_ret_df, min_assets=MIN_ASSETS_PER_DATE):
    """Cross-sectional Spearman (rank) IC per date."""
    rows = []
    for dt in factor_df.index:
        f = factor_df.loc[dt].dropna()
        r = fwd_ret_df.loc[dt].reindex(f.index).dropna()
        common = f.index.intersection(r.index)
        if len(common) < min_assets:
            continue
        ic = f[common].rank().corr(r[common].rank())
        if np.isfinite(ic):
            rows.append((dt, ic, len(common)))
    if not rows:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    s = pd.DataFrame(rows, columns=['date', 'ic', 'n']).set_index('date')
    return s['ic'], s['n']


def summarize(ic_series, n_series, name, fwd=None, factor_df=None, label=''):
    """Produce summary metrics for a factor's IC series."""
    ic = ic_series.dropna()
    out = {'factor': name}
    if len(ic) == 0:
        out.update({'ic': np.nan, 'icir': np.nan, 'hit': np.nan,
                    'n_dates': 0, 'status': 'NO_DATA'})
        return out
    out.update({
        'ic': float(ic.mean()),
        'icir': float(ic.mean() / ic.std(ddof=1)) if ic.std(ddof=1) > 0 else 0.0,
        'hit': float((ic > 0).mean()),
        'n_dates': int(len(ic)),
        'mean_n_assets': float(n_series.reindex(ic.index).mean()),
    })
    for lo, hi, tag in [('2020-01-01', '2022-12-31', 'r2020_22'),
                        ('2023-01-01', '2026-12-31', 'r2023_26')]:
        sub = ic[(ic.index >= lo) & (ic.index <= hi)]
        out[f'{tag}_ic'] = float(sub.mean()) if len(sub) else np.nan
        out[f'{tag}_n'] = int(len(sub))
    if fwd is not None and factor_df is not None:
        dec = {}
        for h in sorted(fwd.keys()):
            ic_h, _ = rank_ic_series(factor_df, fwd[h])
            ic_h = ic_h.dropna()
            dec[str(h)] = float(ic_h.mean()) if len(ic_h) else np.nan
        out['decay_ic'] = dec
    if factor_df is not None and len(factor_df) > 2:
        rk = factor_df.rank(axis=1)
        chg = rk.diff().abs().mean(axis=1).dropna()
        out['turnover_rank_abs'] = float(chg.mean()) if len(chg) else np.nan
    cov = factor_df.notna().mean()
    out['coverage_asset_days'] = float(cov.mean())
    out['coverage_dates_ge8'] = float((n_series >= 8).mean()) if len(n_series) else np.nan
    out['label'] = label
    return out


def library_factor_signals(panel):
    """Signals of the 4 existing effective factors (for correlation audit)."""
    close = panel
    sig = {}
    sig['mom_10d_skip5'] = close.shift(5) / close.shift(15) - 1.0
    sig['mom_120d_skip5'] = close.shift(5) / close.shift(125) - 1.0
    ret = close.pct_change()
    sig['vol_of_vol20x60'] = ret.rolling(20).std().rolling(60).std()
    macro = load_macro_panel()
    vixr = macro['VIX'].pct_change()
    beta = ret.rolling(60).cov(vixr) / vixr.rolling(60).var()
    sig['vix_beta_cond_60x20'] = -beta * (macro['VIX'] / macro['VIX'].shift(20) - 1.0)
    return sig


def max_abs_library_corr(candidate_df, library_sigs, common_index=None):
    """Max |pairwise rank corr| of candidate signal vs each library signal."""
    if common_index is not None:
        candidate_df = candidate_df.reindex(common_index)
    best = 0.0
    details = {}
    for name, sig in library_sigs.items():
        both = pd.concat([candidate_df.stack().rename('cand'),
                          sig.stack().rename('lib')], axis=1).dropna()
        if len(both) < 30:
            continue
        rho = both['cand'].rank().corr(both['lib'].rank())
        if not np.isfinite(rho):
            rho = 0.0
        details[name] = float(rho)
        best = max(best, abs(rho))
    return best, details


def admit_gate(ic, icir, max_rho):
    """Apply benchmark admission gates (absolute values)."""
    ic_ok = abs(ic) >= IC_TH
    icir_ok = abs(icir) >= ICIR_TH
    rho_ok = max_rho < 0.5
    return ic_ok and icir_ok and rho_ok, {
        'ic_ok': ic_ok, 'icir_ok': icir_ok, 'rho_ok': rho_ok,
        'ic': float(ic), 'icir': float(icir), 'max_rho': float(max_rho),
    }
