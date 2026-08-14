"""miner_1 2034-07-20 novel factor screening (data through 2034-07-19).

Tests a batch of interpretable cross-asset candidates: mean-reversion/crash-recovery,
risk-adjusted trend, skew, overextension reversal, co-movement regime, macro-conditional
betas (DXY/VIX/crypto/copper-wti). Reports full-sample / trailing-365d / trailing-120d
IC and ICIR at horizon 10, coverage, 10d rank turnover, and max |corr| vs the 5
ensemble factors (spx_corr60, max_consec_gain_20, downbeta_spx_60, mom_180d_skip5,
range_pos_252) as max_abs_library_correlation provenance.
"""
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

ASSETS = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
          'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
HORIZON = 10
MIN_ASSETS = 8
GATE_IC = 0.0070
GATE_ICIR = 0.0840
ENSEMBLE_FIDS = ['spx_corr60', 'max_consec_gain_20', 'downbeta_spx_60', 'mom_180d_skip5', 'range_pos_252']

def load_close():
    out = {}
    for s in ASSETS:
        df = get_stock_daily_data(symbol=s, days=4000)
        if df is None or len(df) < 300:
            continue
        df = df.copy(); df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        out[s] = df['close'].astype(float)
    idx = None
    for s, ser in out.items():
        idx = ser.index if idx is None else idx.union(ser.index)
    idx = idx.sort_values()
    for s in out:
        out[s] = out[s].reindex(idx)
    return pd.DataFrame(out)

def load_macro(name, C):
    df = pd.read_csv(f'../persistent/index_data/{name}.csv', parse_dates=['date'])
    df = df.set_index('date')['close'].astype(float)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df.reindex(C.index).ffill()

def ic_series(factor_df, fwd_df):
    ics, dates = [], []
    for dt in factor_df.index:
        x = factor_df.loc[dt]; y = fwd_df.loc[dt]
        m = x.notna() & y.notna()
        if m.sum() >= MIN_ASSETS:
            v = x[m].rank().corr(y[m].rank())
            if np.isfinite(v):
                ics.append(v); dates.append(dt)
    return pd.Series(ics, index=pd.DatetimeIndex(dates))

def summ(ic_s):
    if len(ic_s) < 5:
        return None
    ic = float(ic_s.mean()); icir = float(ic_s.mean()/ic_s.std()) if ic_s.std() > 0 else np.nan
    hit = float((ic_s > 0).mean())
    return {'ic': round(ic, 4), 'icir': round(icir, 3), 'hit': round(hit, 3), 'n': len(ic_s),
            'pass': bool(np.isfinite(ic) and np.isfinite(icir) and abs(ic) >= GATE_IC and abs(icir) >= GATE_ICIR)}

def rank_turnover(panel):
    """mean |rank change| over 10d rebalance steps, normalized by n-1."""
    r = panel.rank(axis=1)
    d = (r - r.shift(10)).abs().mean(axis=1).dropna()
    d = d[d < 1e6]
    return float(d.mean() / (panel.shape[1] - 1)) if len(d) else np.nan

def corr_with_library(panel, C, fwd):
    """max |spearman corr| of candidate vs library factors (evaluable expressions + ensemble)."""
    lib = {}
    for fid in ENSEMBLE_FIDS:
        try:
            d = json.load(open(f'factors/{fid}.json'))
            expr = d['calculation']['expression'].replace('close', 'C')
            p = eval(expr, {'__builtins__': {}}, {'C': C, 'np': np, 'pd': pd})
            if isinstance(p, pd.Series):
                p = p.to_frame(C.columns[0])
            lib[fid] = p.reindex(columns=C.columns).astype(float)
        except Exception:
            pass
    maxc = 0.0; arg = None
    for fid, p in lib.items():
        m = panel.notna() & p.notna() & fwd.notna()
        # time-series avg of cross-sectional corr
        cs = []
        for dt in panel.index:
            x = panel.loc[dt].rank(); y = p.loc[dt].rank()
            mm = x.notna() & y.notna() & fwd.loc[dt].notna()
            if mm.sum() >= 6:
                v = x[mm].corr(y[mm])
                if np.isfinite(v):
                    cs.append(v)
        if cs:
            v = float(np.mean(cs))
            if abs(v) > maxc:
                maxc = abs(v); arg = fid
    return maxc, arg

def main():
    C = load_close()
    DXY = load_macro('DXY', C); USDJPY = load_macro('USDJPY', C)
    EURUSD = load_macro('EURUSD', C); VIX = load_macro('VIX', C); USDCNY = load_macro('USDCNY', C)
    print('grid', C.shape, C.index.min().date(), '->', C.index.max().date())
    ret = C.pct_change()
    fwd = C.shift(-HORIZON) / C - 1.0

    def roll_std(x, w, minp=20):
        return x.rolling(w, min_periods=minp).std()

    def roll_beta(asset_ret, mkt_ret, w, minp):
        cov = asset_ret.rolling(w, min_periods=minp).cov(mkt_ret)
        var = mkt_ret.rolling(w, min_periods=minp).var()
        return cov / var

    cands = {}
    # 1. crash_bounce_60: mean reversion in crash legs (negative 60d ret predicts bounce)
    r60 = ret.rolling(60, min_periods=40).sum()
    v60 = roll_std(ret, 60)
    cands['crash_bounce_60'] = -1.0 * (r60 / v60) * (r60 < 0).astype(float)
    # 2. risk_adj_trend_20
    r20 = ret.rolling(20, min_periods=15).sum()
    v20 = roll_std(ret, 20, 15)
    cands['risk_adj_trend_20'] = r20 / v20
    # 3. skew_60 (crash-risk: negative skew)
    cands['skew_60'] = ret.rolling(60, min_periods=40).skew()
    # 4. bollinger_rev_20: overextension reversal
    sma20 = C.rolling(20, min_periods=15).mean()
    sd20 = C.rolling(20, min_periods=15).std()
    cands['bollinger_rev_20'] = -1.0 * ((C - sma20) / (2 * sd20)).clip(-1.5, 1.5)
    # 5. stoch_osc_20 reversal (high stochastic -> sell)
    lo20 = C.rolling(20, min_periods=15).min(); hi20 = C.rolling(20, min_periods=15).max()
    cands['stoch_rev_20'] = -1.0 * ((C - lo20) / (hi20 - lo20) - 0.5)
    # 6. cross_med_corr_60: co-movement regime (avg corr of asset ret to cross-sectional median ret)
    med_ret = ret.median(axis=1)
    cm = pd.DataFrame(index=C.index, columns=C.columns, dtype=float)
    for s in C.columns:
        cm[s] = ret[s].rolling(60, min_periods=40).corr(med_ret)
    cands['cross_med_corr_60'] = cm
    # 7. dxy_trend_beta_60x20: dollar-trend conditional beta
    dxy_r = DXY.pct_change()
    b_dxy = roll_beta(ret, dxy_r, 60, 40)
    cands['dxy_trend_beta_60x20'] = b_dxy * (DXY / DXY.shift(20) - 1.0)
    # 8. vix_level_beta_60: vol-regime conditional beta (risk-off transmission)
    vix_r = VIX.pct_change()
    b_vix = roll_beta(ret, vix_r, 60, 40)
    cands['vix_level_beta_60'] = b_vix * (VIX / VIX.shift(20) - 1.0)
    # 9. crypto_beta_trend: risk-appetite transmission via ETH
    eth_r = ret['ETH']
    b_eth = roll_beta(ret, eth_r, 60, 40)
    cands['crypto_beta_trend_60x20'] = b_eth * (C['ETH'] / C['ETH'].shift(20) - 1.0)
    # 10. drawdown_vol_60: risk-adjusted drawdown depth
    hi60 = C.rolling(60, min_periods=40).max()
    cands['drawdown_vol_60'] = (C / hi60 - 1.0) / v60
    # 11. copper_wti_ratio_beta: cyclical spread transmission
    cw = C['COPPER'] / C['WTI']
    cw_r = cw.pct_change()
    b_cw = roll_beta(ret, cw_r, 60, 40)
    cands['copper_wti_beta_60x20'] = b_cw * (cw / cw.shift(20) - 1.0)
    # 12. usd_carry_regime: USDCNY trend conditional on EURUSD (EM/fx pressure)
    eur_r = EURUSD.pct_change()
    b_eur = roll_beta(ret, eur_r, 60, 40)
    cands['eurusd_beta_cny_trend'] = b_eur * (USDCNY / USDCNY.shift(20) - 1.0)

    print(f"{'factor':26s} {'FULL ic/icir':>16s} {'365d ic/icir':>15s} {'120d ic/icir':>15s} {'cov':>6s} {'to10':>6s} {'max|rho|':>9s} {'vs':>14s}")
    results = {}
    for fid, panel in cands.items():
        panel = panel.reindex(columns=C.columns).astype(float)
        ic_s = ic_series(panel, fwd)
        full = summ(ic_s); last365 = summ(ic_s[ic_s.index >= ic_s.index[-1] - pd.Timedelta(days=365)]) if len(ic_s) else None
        last120 = summ(ic_s[ic_s.index >= ic_s.index[-1] - pd.Timedelta(days=120)]) if len(ic_s) else None
        cov = float(panel.notna().mean().mean())
        to = rank_turnover(panel)
        mc, arg = corr_with_library(panel, C, fwd)
        results[fid] = dict(full=full, last365=last365, last120=last120, cov=cov, to=to, mc=mc, arg=arg)
        f = full or {}; a = last365 or {}; b = last120 or {}
        print(f"{fid:26s} {f.get('ic',float('nan')):+.4f}/{f.get('icir',float('nan')):+.3f}{'*' if f.get('pass') else ' '} "
              f"{a.get('ic',float('nan')):+.4f}/{a.get('icir',float('nan')):+.3f} "
              f"{b.get('ic',float('nan')):+.4f}/{b.get('icir',float('nan')):+.3f} {cov:.3f} {to:.3f} {mc:.3f} {arg}")
    print('\n=== CANDIDATES PASSING FULL GATE (|ic|>=0.0070 & |icir|>=0.0840) ===')
    for fid, r in results.items():
        if r['full'] and r['full']['pass']:
            print(f"{fid:26s} ic={r['full']['ic']:+.4f} icir={r['full']['icir']:+.3f} hit={r['full']['hit']:.3f} "
                  f"n={r['full']['n']} 365d={r['last365']['ic'] if r['last365'] else float('nan'):+.4f} "
                  f"120d={r['last120']['ic'] if r['last120'] else float('nan'):+.4f} cov={r['cov']:.3f} to10={r['to']:.3f} maxrho={r['mc']:.3f} vs {r['arg']}")

if __name__ == '__main__':
    main()
