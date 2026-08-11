"""Shared validation helper for miner_3 factor research (2026-07-30 cycle)."""
import sys, os, json, math, zlib, base64, hashlib
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
from alphacrafter.sim.utils import get_stock_daily_data

WATCHLIST = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
             'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
LIB_FACTORS = ['trend_r2_30_signed', 'semi_down_ratio_20', 'mom_120d_skip5',
               'mom_10d_skip5', 'time_under_water_120', 'vol_of_vol20x60',
               'dxy_beta_60', 'WTI_BETA_60', 'vix_beta_cond_60x20']


def load_ohlcv(symbol, days=4000):
    df = get_stock_daily_data(symbol=symbol, days=days)
    if df is None or len(df) == 0:
        return None
    df = df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.set_index('date').sort_index()
    return df


def load_close_panel(days=4000):
    closes, vols, highs, lows, opens = {}, {}, {}, {}, {}
    for s in WATCHLIST:
        df = load_ohlcv(s, days)
        closes[s] = df['close']
        vols[s] = df['volume']
        highs[s] = df['high']
        lows[s] = df['low']
        opens[s] = df['open']
    C = pd.DataFrame(closes).sort_index()
    V = pd.DataFrame(vols).sort_index()
    H = pd.DataFrame(highs).sort_index()
    L = pd.DataFrame(lows).sort_index()
    O = pd.DataFrame(opens).sort_index()
    return C, V, H, L, O


def rank_ic(factor_panel, fwd_ret_panel):
    """Daily cross-sectional Spearman IC between factor and forward returns."""
    dates = factor_panel.index.intersection(fwd_ret_panel.index)
    ics = []
    for d in dates:
        f = factor_panel.loc[d]
        r = fwd_ret_panel.loc[d]
        m = f.notna() & r.notna()
        if m.sum() >= 8:
            ic = f[m].rank().corr(r[m].rank())
            if not math.isnan(ic):
                ics.append((d, ic))
    if not ics:
        return None
    s = pd.Series([x[1] for x in ics], index=[x[0] for x in ics], dtype=float)
    return s


def summarize(s, horizon, label=""):
    ic = s.mean()
    icir = s.mean() / s.std() if s.std() > 0 else 0.0
    hit = (s > 0).mean()
    n = len(s)
    regime = {}
    for name, lo, hi in [("2020-2022", "2020-01-01", "2022-12-31"),
                          ("2023-2024", "2023-01-01", "2024-12-31"),
                          ("2025-2026", "2025-01-01", "2026-12-31")]:
        sub = s[(s.index >= lo) & (s.index <= hi)]
        if len(sub) >= 20:
            regime[name] = {"ic": round(sub.mean(), 4), "icir": round(sub.mean() / sub.std(), 4) if sub.std() > 0 else 0.0, "n": int(len(sub))}
    return {"label": label, "horizon": horizon, "ic": round(ic, 4), "icir": round(icir, 4),
            "ic_hit_ratio": round(hit, 3), "n_ic_dates": int(n), "regime": regime}


def decay_analysis(factor_panel, ret_panel, horizons=(1, 2, 3, 5, 10, 20)):
    out = {}
    for h in horizons:
        fr = ret_panel.shift(-h)
        s = rank_ic(factor_panel, fr)
        if s is not None and len(s) >= 20:
            out[str(h)] = round(s.mean(), 4)
    return out


def coverage_turnover(factor_panel, ret_panel, horizon=10, every=10):
    valid = factor_panel.notna()
    cov_asset_days = float(valid.sum().sum()) / float(valid.size)
    ge8 = valid.sum(axis=1) >= 8
    cov_dates_ge8 = float(ge8.mean())
    # rank turnover at decision frequency (every 10d)
    to = []
    idx = factor_panel.index[::every]
    for i in range(1, len(idx)):
        d0, d1 = idx[i - 1], idx[i]
        r0 = factor_panel.loc[d0].rank()
        r1 = factor_panel.loc[d1].rank()
        m = r0.notna() & r1.notna()
        if m.sum() >= 8:
            to.append((r1[m] - r0[m]).abs().mean())
    turnover = float(np.mean(to)) if to else float('nan')
    return {"coverage_asset_days": round(cov_asset_days, 3),
            "coverage_dates_ge8": round(cov_dates_ge8, 3),
            "turnover_10d_rank": round(turnover, 3)}


def decode_artifact(artifact):
    raw = base64.b64decode(artifact['data'])
    csv_txt = zlib.decompress(raw).decode('utf-8')
    df = pd.read_csv(pd.io.common.StringIO(csv_txt), index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def library_max_rho(new_panel, lib_dir='factors'):
    """Flattened Pearson rho between new factor panel and each library signal artifact."""
    rhos = {}
    new_flat = new_panel.stack()
    for fid in LIB_FACTORS:
        p = os.path.join(lib_dir, fid + '.json')
        if not os.path.exists(p):
            continue
        try:
            d = json.load(open(p))
            art = d.get('validation', {}).get('signal_artifact')
            if not art:
                rhos[fid] = None
                continue
            libp = decode_artifact(art)
            common = new_panel.index.intersection(libp.index)
            a = new_panel.loc[common].stack()
            b = libp.loc[common].stack()
            m = a.notna() & b.notna()
            if m.sum() >= 200:
                r = float(np.corrcoef(a[m], b[m])[0, 1])
                rhos[fid] = round(r, 3) if not math.isnan(r) else None
            else:
                rhos[fid] = None
        except Exception as e:
            rhos[fid] = None
    vals = [v for v in rhos.values() if v is not None]
    return rhos, (max(vals) if vals else 0.0)


def build_artifact(panel):
    df = panel.copy()
    df.index = df.index.strftime('%Y-%m-%d')
    csv_txt = df.to_csv()
    comp = zlib.compress(csv_txt.encode('utf-8'))
    b64 = base64.b64encode(comp).decode('ascii')
    sha = hashlib.sha256(b64.encode('ascii')).hexdigest()
    return {
        "format": "base64:zlib:csv",
        "description": "Factor signal panel: rows = dates (YYYY-MM-DD), cols = 15 watchlist symbols. Recover with zlib.decompress(base64.b64decode(data)).decode() then pd.read_csv(StringIO).",
        "columns": list(panel.columns),
        "shape": [int(panel.shape[0]), int(panel.shape[1])],
        "n_valid_values": int(panel.notna().sum().sum()),
        "sha256": sha,
        "data": b64,
    }


def full_validate(factor_panel, ret_panel, horizon=10, label=""):
    s = rank_ic(factor_panel, ret_panel.shift(-horizon))
    summ = summarize(s, horizon, label)
    summ['decay_ic_by_horizon'] = decay_analysis(factor_panel, ret_panel)
    cov = coverage_turnover(factor_panel, ret_panel, horizon)
    summ.update(cov)
    rhos, maxrho = library_max_rho(factor_panel)
    summ['library_rho_by_factor'] = rhos
    summ['max_abs_library_correlation'] = round(maxrho, 3)
    return summ
