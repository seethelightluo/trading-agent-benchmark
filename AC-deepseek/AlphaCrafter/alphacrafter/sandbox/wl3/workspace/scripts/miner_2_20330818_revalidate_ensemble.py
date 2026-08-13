"""miner_2 2033-08-18: re-validate the 10 ensemble factors on (a) warm-up admission
window 2020-01-01..2026-07-15, (b) recent 2y window 2031-08-18..2033-08-17,
(c) full sample 2020-01-01..2033-08-17. Checks drift and timeliness."""
import sys, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import load_prices, load_index, WATCHLIST, VAL_START, VAL_END

t0 = time.time()
prices = load_prices(days=3400)
print(f"assets loaded: {len(prices)}; max date: {max(d.index.max() for d in prices.values())} ({time.time()-t0:.1f}s)")

dxy = load_index('DXY', days=3400, prices=prices)
print("DXY loaded:", dxy is not None, dxy.index.max() if dxy is not None else None)

ret_wide = pd.DataFrame({s: prices[s]['close'].pct_change() for s in WATCHLIST}).sort_index()
spx_ret = ret_wide['SPX']
cn10y = prices['CN10Y']['close']
cn10y_chg = cn10y.diff()
comm_ret = ret_wide[['XAU', 'COPPER', 'WTI']].mean(axis=1)
dxy_ret = dxy['close'].pct_change() if dxy is not None else None

def rolling_beta(y, x, w):
    z = pd.concat([y.rename('y'), x.rename('x')], axis=1).dropna()
    cov = z['y'].rolling(w, min_periods=int(w*0.6)).cov(z['x'])
    var = z['x'].rolling(w, min_periods=int(w*0.6)).var()
    b = (cov / var.replace(0, np.nan)).reindex(z.index)
    return b

FACTORS = {}

def f_down_beta(df, s):
    r = df['close'].pct_change()
    m = spx_ret < 0
    z = pd.concat([r.rename('r'), spx_ret.rename('s')], axis=1)
    zd = z[m]
    cov = zd['r'].rolling(60, min_periods=36).cov(zd['s'])
    var = zd['s'].rolling(60, min_periods=36).var()
    b = (cov / var.replace(0, np.nan)).reindex(z.index)
    return b
FACTORS['down_beta_60'] = f_down_beta

def f_cn10y_beta(df, s):
    return rolling_beta(df['close'].pct_change(), cn10y_chg, 60)
FACTORS['cn10y_beta_60'] = f_cn10y_beta

def f_spx_beta(df, s):
    return rolling_beta(df['close'].pct_change(), spx_ret, 60)
FACTORS['spx_beta_60'] = f_spx_beta

def f_vol_adj_mom(df, s):
    r = df['close'].pct_change()
    mom = df['close'].shift(5) / df['close'].shift(25) - 1.0
    vol = r.rolling(60, min_periods=40).std()
    return mom / vol.replace(0, np.nan)
FACTORS['vol_adj_mom_20_60'] = f_vol_adj_mom

def f_comm_basket_beta(df, s):
    return rolling_beta(df['close'].pct_change(), comm_ret, 60)
FACTORS['comm_basket_beta_60'] = f_comm_basket_beta

def f_hs300_beta(df, s):
    return rolling_beta(df['close'].pct_change(), ret_wide['000300.SH'], 60)
FACTORS['hs300_beta_60'] = f_hs300_beta

def f_intraday_skew(df, s):
    ir = df['close'] / df['open'] - 1.0
    return ir.rolling(20, min_periods=12).skew()
FACTORS['intraday_ret_skew_20'] = f_intraday_skew

def f_vov(df, s):
    return df['close'].pct_change().rolling(20, min_periods=12).std().rolling(60, min_periods=30).std()
FACTORS['vol_of_vol20x60'] = f_vov

def f_dxy_beta_cond(df, s):
    b = rolling_beta(df['close'].pct_change(), dxy_ret, 60)
    dxy_mom = dxy['close'] / dxy['close'].shift(20) - 1.0
    return (b * dxy_mom).reindex(b.index)
FACTORS['dxy_beta_cond_60x20'] = f_dxy_beta_cond

def f_hilo_vol_ratio(df, s):
    hi = df['close'].rolling(20, min_periods=12).max()
    lo = df['close'].rolling(20, min_periods=12).min()
    rng = (hi - lo) / df['close']
    vol = df['close'].pct_change().rolling(20, min_periods=12).std()
    return rng / vol.replace(0, np.nan)
FACTORS['hilo_vol_ratio_20'] = f_hilo_vol_ratio


def factor_to_panel(fn, prices):
    cols = {}
    for s, df in prices.items():
        try:
            ser = fn(df, s)
            if ser is not None and len(ser) > 0:
                cols[s] = ser.astype(float)
        except Exception as e:
            pass
    panel = pd.DataFrame(cols)
    panel = panel[~panel.index.duplicated(keep='last')].sort_index()
    return panel


def fwd_ret(h):
    cols = {}
    for s, df in prices.items():
        cols[s] = df['close'].shift(-h) / df['close'] - 1.0
    return pd.DataFrame(cols).sort_index()

FWD = {h: fwd_ret(h) for h in (1, 2, 3, 5, 10, 20)}

def rank_ic_series(fac, fwd, min_valid=8):
    common = fac.index.intersection(fwd.index)
    ic = {}
    for d in common:
        x = fac.loc[d]; y = fwd.loc[d]
        m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if m.sum() >= min_valid:
            ic[d] = x[m].rank().corr(y[m].rank())
    return pd.Series(ic).sort_index()

def validate_window(fac, start, end):
    ic10 = rank_ic_series(fac, FWD[10])
    ic10 = ic10[(ic10.index >= start) & (ic10.index <= end)]
    if len(ic10) < 60:
        return None
    icm = float(ic10.mean()); ics = float(ic10.std(ddof=1))
    icir = icm / ics if ics > 0 else 0.0
    hit = float((ic10 > 0).mean()) if icm >= 0 else float((ic10 < 0).mean())
    f = fac[(fac.index >= start) & (fac.index <= end)]
    cov = float(f.notna().sum().sum()) / (f.shape[0]*f.shape[1]) if f.shape[0]*f.shape[1] else 0
    ge8 = float((f.notna().sum(axis=1) >= 8).mean())
    decay = {}
    for h in (1,2,3,5,10,20):
        ic = rank_ic_series(fac, FWD[h])
        ic = ic[(ic.index >= start) & (ic.index <= end)]
        decay[str(h)] = float(ic.mean()) if len(ic) else float('nan')
    return {'ic': icm, 'icir': icir, 'ic_hit_ratio': hit, 'n_ic_dates': int(len(ic10)),
            'coverage_asset_days': cov, 'coverage_dates_ge8': ge8, 'decay_ic_by_horizon': decay}

WARM = (VAL_START, VAL_END)
RECENT = (pd.Timestamp('2031-08-18'), pd.Timestamp('2033-08-17'))
FULL = (pd.Timestamp('2020-01-01'), pd.Timestamp('2033-08-17'))

print(f"\n{'factor':<22}{'warm_ic':>9}{'warm_icir':>10}{'rec_ic':>9}{'rec_icir':>10}{'full_ic':>9}{'full_icir':>10}")
for fid, fn in FACTORS.items():
    fac = factor_to_panel(fn, prices)
    mw = validate_window(fac, *WARM)
    mr = validate_window(fac, *RECENT)
    mf = validate_window(fac, *FULL)
    if mw is None:
        print(f"{fid:<22}  insufficient"); continue
    w_ic, w_icir = mw['ic'], mw['icir']
    r_ic = mr['ic'] if mr else float('nan')
    r_icir = mr['icir'] if mr else float('nan')
    f_ic = mf['ic'] if mf else float('nan')
    f_icir = mf['icir'] if mf else float('nan')
    flag = ''
    if mr and abs(mr['ic']) < 0.007: flag += ' RECENT-IC-WEAK'
    if mr and mr['icir'] < 0 and mw['ic'] > 0: flag += ' RECENT-ICIR-NEG'
    if mr and mw['ic'] > 0 and mr['ic'] < -0.5*mw['ic']: flag += ' DRIFT-SIGN'
    print(f"{fid:<22}{w_ic:>9.4f}{w_icir:>10.4f}{r_ic:>9.4f}{r_icir:>10.4f}{f_ic:>9.4f}{f_icir:>10.4f}{flag}")
    # save details for later
    with open(f'scripts/miner_2_20330818_reval_{fid}.json', 'w') as fh:
        json.dump({'warm': mw, 'recent': mr, 'full': mf}, fh, default=str, indent=1)

print(f"\ndone in {time.time()-t0:.1f}s")
