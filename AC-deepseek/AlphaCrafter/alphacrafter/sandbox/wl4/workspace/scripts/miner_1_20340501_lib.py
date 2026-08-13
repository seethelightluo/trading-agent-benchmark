"""Shared validation harness for miner_1 (2034-05-01 cycle).

Loads the 15-asset tradable panel plus observation-only macro signals, computes
a candidate factor panel, and evaluates cross-sectional rank IC at multiple
horizons. Data is strictly capped at 2034-04-28 (last completed trading day
before the current date 2034-05-01). Macro CSVs in ../persistent/index_data/
contain synthetic rows beyond the current date -> sliced to <= 2034-04-28.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MACRO = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']
MAX_DATE = pd.Timestamp('2034-04-28')


def load_close_panel(min_obs: int = 100):
    """Load close-price panel (rows=dates, cols=assets) capped at MAX_DATE."""
    frames = {}
    for s in WATCH:
        df = get_stock_daily_data(symbol=s, days=4200)
        if df is None:
            continue
        df = df[df['date'] <= MAX_DATE].copy()
        df = df.set_index('date')
        frames[s] = df['close']
    panel = pd.DataFrame(frames).sort_index()
    # drop leading dates with < min_obs assets
    panel = panel.dropna(axis=0, how='all')
    return panel


def load_ohlc_panel():
    """Load OHLCV panel (MultiIndex cols) capped at MAX_DATE."""
    frames = {}
    for s in WATCH:
        df = get_stock_daily_data(symbol=s, days=4200)
        if df is None:
            continue
        df = df[df['date'] <= MAX_DATE].copy()
        df = df.set_index('date')
        frames[s] = df[['open', 'high', 'low', 'close', 'volume']]
    panel = pd.concat(frames, axis=1).sort_index()
    return panel


def load_macro():
    """Load macro observation series capped at MAX_DATE."""
    out = {}
    for s in MACRO:
        df = pd.read_csv(f'../persistent/index_data/{s}.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] <= MAX_DATE].set_index('date')
        out[s] = df['close']
    return pd.DataFrame(out).sort_index()


def rets(panel):
    return panel.pct_change(fill_method=None)


def forward_return(panel, h: int):
    """Forward h-day return: fwd_ret_t = close_{t+h}/close_t - 1."""
    return panel.shift(-h) / panel - 1.0


def rank_ic_series(factor_panel, fwd):
    """Daily cross-sectional Spearman rank IC between factor and fwd return.

    A date qualifies if >= 8 assets have both valid factor and fwd values.
    Returns (ic_series, n_dates, coverage_asset_days, coverage_dates_ge8).
    """
    dates = factor_panel.index.intersection(fwd.index)
    ic = {}
    n_valid_dates = 0
    total_obs = 0
    valid_obs = 0
    for d in dates:
        f = factor_panel.loc[d]
        r = fwd.loc[d]
        mask = f.notna() & r.notna() & np.isfinite(f.values.astype(float)) & np.isfinite(r.values.astype(float))
        if mask.sum() < 8:
            continue
        total_obs += len(f)
        valid_obs += int(mask.sum())
        n_valid_dates += 1
        ic[d] = f[mask].rank().corr(r[mask].rank())
    s = pd.Series(ic, dtype=float)
    cov_dates = (s.notna()).mean() if len(s) else 0.0
    cov_asset = valid_obs / total_obs if total_obs else 0.0
    return s, n_valid_dates, cov_asset, cov_dates


def evaluate_factor(factor_panel, horizons=(1, 2, 3, 5, 10, 20), min_dates=200,
                    verbose=True):
    """Evaluate a factor panel end-to-end. Returns dict of metrics."""
    panel = load_close_panel()
    # align factor panel to close panel dates
    factor_panel = factor_panel.reindex(panel.index)
    out = {}
    for h in horizons:
        fwd = forward_return(panel, h)
        ic_series, n_dates, cov_asset, cov_dates = rank_ic_series(factor_panel, fwd)
        if len(ic_series) == 0:
            out[h] = {'ic': np.nan, 'icir': np.nan, 'hit': np.nan,
                      'n_dates': 0, 'cov_asset': 0.0, 'cov_dates': 0.0}
            continue
        ic = float(ic_series.mean())
        icstd = float(ic_series.std(ddof=1))
        icir = ic / icstd * np.sqrt(len(ic_series)) if icstd > 0 else np.nan
        hit = float((ic_series > 0).mean())
        out[h] = {'ic': ic, 'icir': icir, 'hit': hit, 'n_dates': n_dates,
                  'cov_asset': cov_asset, 'cov_dates': cov_dates, 'ic_std': icstd}
        if verbose:
            print(f'  h={h:>2d}: IC={ic:+.4f}  ICIR={icir:+.3f}  hit={hit:.3f}  '
                  f'n_dates={n_dates}  cov_asset={cov_asset:.3f}  cov_dates={cov_dates:.3f}')
    return out


def turnover_10d_rank(factor_panel):
    """Average rank change over 10-day lags (lower = more stable)."""
    r = factor_panel.rank(axis=1)
    d = r.diff(10).abs().mean(skipna=True)
    return float(d.mean())


def load_library_signals():
    """Load signal artifacts of active library factors into a dict of Series panels."""
    import json, base64, zlib, io
    out = {}
    for fid in ['vol_adj_mom_accel_20x60', 'dn_mkt_beta_60d', 'rate_beta_cn10y_60d']:
        try:
            d = json.load(open(f'factors/{fid}.json'))
            sa = d['validation']['signal_artifact']
            fmt = sa.get('format', '')
            if fmt == 'panel_json_v1':
                dates = pd.to_datetime(sa['dates'])
                assets = sa['assets']
                values = sa['values']
                arr = np.full((len(dates), len(assets)), np.nan)
                for j, a in enumerate(assets):
                    arr[:, j] = np.array([np.nan if v is None else v for v in values[a]], dtype=float)
                df = pd.DataFrame(arr, index=dates, columns=assets)
            elif fmt == 'base64:zlib:csv':
                raw = base64.b64decode(sa['data'])
                csv = zlib.decompress(raw).decode()
                df = pd.read_csv(io.StringIO(csv), index_col=0, parse_dates=True)
                df.columns = sa.get('columns', df.columns)
            else:
                continue
            out[fid] = df
        except Exception as e:
            print(f'  [warn] could not load {fid}: {e}')
    return out


def max_lib_corr(factor_panel, lib_signals, common_dates=None):
    """Max |Spearman rho| of candidate factor vs library factor panels (same date)."""
    out = {}
    for fid, lib in lib_signals.items():
        f = factor_panel.reindex(lib.index)
        # cross-sectional values flattened by date: compare per-date ranks concatenated
        rs = []
        for d in f.index.intersection(lib.index):
            a = f.loc[d].astype(float)
            b = lib.loc[d].astype(float)
            m = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
            if m.sum() < 8:
                continue
            rs.append(pd.Series({'a': a[m].rank().mean(), 'b': b[m].rank().mean()}) if False else
                      (a[m].rank().corr(b[m].rank())))
        if len(rs) == 0:
            continue
        daily_corrs = [r for r in rs if r is not None and np.isfinite(r)]
        if len(daily_corrs) == 0:
            continue
        mean_rho = float(np.mean(daily_corrs))
        out[fid] = {'mean_daily_rho': mean_rho, 'n_dates': len(daily_corrs)}
    if not out:
        return 0.0, None
    maxf = max(out, key=lambda k: abs(out[k]['mean_daily_rho']))
    return abs(out[maxf]['mean_daily_rho']), maxf


if __name__ == '__main__':
    p = load_close_panel()
    print('close panel shape:', p.shape, 'first:', p.index[0].date(), 'last:', p.index[-1].date())
    m = load_macro()
    print('macro shape:', m.shape, 'last:', m.index[-1].date())
