"""Shared factor validation library for miner_1 (and other miners).

Loads the 15-asset tradable universe through the API (no lookahead: data ends
at the simulator's visible date), computes rank IC/ICIR vs forward returns,
decay, turnover, coverage, and max abs library correlation vs persisted
EFFECTIVE factor signal artifacts.

Admission gates (benchmark-wide, 15-asset universe):
  abs(IC)  >= 0.0070
  abs(ICIR) >= 0.0840   where ICIR = mean(IC)/std(IC)
"""
import base64
import zlib
import json
import glob
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

WATCHLIST = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX',
             'NDX', 'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

# observation-only macro signals (index_data)
MACRO = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']


def load_asset_panel(days=3000):
    """Return DataFrame of close prices, index=date, cols=assets (15)."""
    frames = {}
    for s in WATCHLIST:
        df = get_stock_daily_data(symbol=s, days=days)
        if df is None or len(df) == 0:
            print(f'WARN: no data for {s}')
            continue
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        frames[s] = df['close']
    panel = pd.DataFrame(frames).sort_index()
    print(f'[load] panel dates {panel.index.min().date()}..{panel.index.max().date()} rows={len(panel)} assets={panel.shape[1]}')
    return panel


def load_macro_panel(days=3000):
    """Return DataFrame of close prices for observation-only macro series."""
    frames = {}
    for s in MACRO:
        df = pd.read_csv(f'../persistent/index_data/{s}.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        frames[s] = df['close']
    panel = pd.DataFrame(frames).sort_index()
    # restrict to simulator visible window (same as API max date)
    panel = panel[panel.index <= pd.Timestamp('2027-07-16')]
    return panel


def fwd_returns(panel, h=10):
    """Forward h-day return per asset: ret_t->t+h, last h rows NaN."""
    return panel.shift(-h) / panel - 1.0


def rank_ic_series(factor, fwd, min_valid=8):
    """Daily cross-sectional Spearman IC between factor and forward return.
    factor, fwd: DataFrames aligned by date/columns.
    Returns Series of IC indexed by date."""
    dates = factor.index.intersection(fwd.index)
    ics = {}
    for dt in dates:
        f = factor.loc[dt]
        r = fwd.loc[dt]
        mask = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        if mask.sum() < min_valid:
            continue
        ic = f[mask].corr(r[mask], method='spearman')
        if np.isfinite(ic):
            ics[dt] = ic
    return pd.Series(ics, name='ic')


def evaluate_factor(factor, panel, h=10, min_valid=8, label='factor',
                    valid_from=None, valid_to=None):
    """Full evaluation: IC, ICIR, hit ratio, decay, turnover, coverage."""
    fwd = fwd_returns(panel, h=h)
    ic = rank_ic_series(factor, fwd, min_valid=min_valid)
    if valid_from is not None:
        ic = ic[ic.index >= pd.Timestamp(valid_from)]
    if valid_to is not None:
        ic = ic[ic.index <= pd.Timestamp(valid_to)]
    out = {'label': label, 'n_ic_dates': len(ic)}
    if len(ic) == 0:
        out.update(ic=np.nan, icir=np.nan, ic_std=np.nan, ic_hit=np.nan)
        return out
    icm = ic.mean()
    ics = ic.std(ddof=1)
    out.update(ic=float(icm), icir=float(icm / ics) if ics > 0 else np.nan,
               ic_std=float(ics), ic_hit=float((ic > 0).mean()))
    # coverage: share of asset-days with valid values
    n_total = len(factor) * factor.shape[1]
    n_valid = int(np.isfinite(factor.values).sum())
    out['coverage_asset_days'] = n_valid / n_total
    ge8 = (factor.notna().sum(axis=1) >= min_valid).mean()
    out['coverage_dates_ge8'] = float(ge8)
    # turnover: mean abs rank change over 10d
    ranks = factor.rank(axis=1)
    turn = (ranks - ranks.shift(10)).abs().mean().mean()
    out['turnover_10d_rank'] = float(turn)
    # decay by horizon
    decay = {}
    for hh in [1, 2, 3, 5, 10, 20]:
        fh = fwd_returns(panel, h=hh)
        ich = rank_ic_series(factor, fh, min_valid=min_valid)
        decay[str(hh)] = float(ich.mean()) if len(ich) else np.nan
    out['decay_ic_by_horizon'] = decay
    return out


def load_library_panels():
    """Decode signal artifacts of all EFFECTIVE factors in factors/*.json.
    Returns dict factor_id -> DataFrame(dates x assets)."""
    panels = {}
    for f in sorted(glob.glob('factors/*.json')):
        if 'ensemble' in f or 'signal' in f:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        v = d.get('validation', {})
        if v.get('status') != 'EFFECTIVE':
            continue
        sa = v.get('signal_artifact')
        if not sa:
            continue
        if sa.get('format') == 'panel_json_v1':
            dates = pd.to_datetime(sa['dates'])
            assets = sa['assets']
            vals = sa['values']
            arr = np.full((len(dates), len(assets)), np.nan)
            for j, a in enumerate(assets):
                arr[:, j] = vals.get(a, [np.nan] * len(dates))
            panels[d['factor_id']] = pd.DataFrame(arr, index=dates, columns=assets)
        else:
            # base64:zlib:csv legacy
            raw = base64.b64decode(sa['data'])
            csv = zlib.decompress(raw).decode()
            pdf = pd.read_csv(pd.io.common.StringIO(csv), index_col=0)
            pdf.index = pd.to_datetime(pdf.index)
            panels[d['factor_id']] = pdf
    return panels


def max_abs_library_correlation(factor, lib_panels, min_valid=8):
    """Per-date cross-sectional Pearson corr between candidate and each
    library factor panel (aligned on overlapping dates); returns max abs over
    factors and dates and the worst factor id."""
    worst = {}
    for fid, lp in lib_panels.items():
        common_dates = factor.index.intersection(lp.index)
        rhos = []
        for dt in common_dates:
            a = factor.loc[dt]
            b = lp.loc[dt]
            m = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
            if m.sum() < min_valid:
                continue
            r = a[m].corr(b[m], method='pearson')
            if np.isfinite(r):
                rhos.append(r)
        if rhos:
            worst[fid] = float(max(abs(r) for r in rhos))
    if not worst:
        return float('nan'), None
    fid = max(worst, key=worst.get)
    return worst[fid], fid


def make_signal_artifact(factor):
    """Serialize factor panel as panel_json_v1 (matches persisted format)."""
    dates = [d.strftime('%Y-%m-%d') for d in factor.index]
    assets = list(factor.columns)
    vals = {}
    for a in assets:
        col = factor[a]
        vals[a] = [None if pd.isna(x) else (None if not np.isfinite(x) else float(x))
                   for x in col]
    return {
        'format': 'panel_json_v1',
        'n_dates': len(dates),
        'n_assets': len(assets),
        'dates': dates,
        'assets': assets,
        'values': vals,
    }


def gates_pass(metrics):
    ic = abs(metrics.get('ic') or 0.0)
    icir = abs(metrics.get('icir') or 0.0)
    return ic >= 0.0070 and icir >= 0.0840, ic, icir


if __name__ == '__main__':
    p = load_asset_panel()
    print(p.tail(2))
