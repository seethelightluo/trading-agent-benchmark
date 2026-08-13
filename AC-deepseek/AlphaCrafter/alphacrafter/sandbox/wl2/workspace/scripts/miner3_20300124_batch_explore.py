"""miner_3 factor exploration batch - 2030-01-24 cycle.
Validate NEW candidate factors on data <= visible_through (2030-01-23). No lookahead.
Admission gates: |IC| >= 0.0070 and |ICIR| >= 0.0840 (15-asset cross-asset universe).

Avoids evicted/explored ideas: carry*, cn10y_corr, cryptobeta_cond, downside_freq,
drawup, eff_ratio, hl_rank, max_gain, mom_curve_volscale, orth_mom20, range_pos_120d,
ret_skew_10, sharpe_20, updown_vol_ratio, vol_price_corr, vol_surge, rev10_volcond,
overnight_20, rsi_14, bb_bandwidth, vol_trend, corr_chg_spx, skew_60, amihud,
beta_asym_spx, maxdd_60, zscore_close_60, accel_5x20, exhaust_20x60, intraday_drift.
"""
import numpy as np
import pandas as pd
import json

VISIBLE = '2030-01-23'
SYMS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
        'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
MIN_ASSETS = 8
HORIZONS = [1, 2, 3, 5, 10, 20]

def load(sym):
    df = pd.read_csv(f'../persistent/stock_data/{sym}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= VISIBLE].reset_index(drop=True)
    for c in ['open','close','high','low','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['ret'] = df['close'].pct_change()
    df['gap'] = df['open'] / df['close'].shift(1) - 1.0
    return df

DATA = {s: load(s) for s in SYMS}

def build_factors(sym):
    df = DATA[sym]
    c = df['close']; o = df['open']; h = df['high']; l = df['low']; r = df['ret']
    vol20 = r.rolling(20, min_periods=10).std()
    vol60 = r.rolling(60, min_periods=15).std()
    ret5 = c / c.shift(5) - 1
    ret20 = c / c.shift(20) - 1
    ret60 = c / c.shift(60) - 1
    out = pd.DataFrame(index=df['date'])
    # 1. Trend consistency: R^2 of linear fit of log-price on time over 60d
    lp = np.log(c)
    def r2_win(x):
        if len(x) < 30 or not np.isfinite(x).all():
            return np.nan
        t = np.arange(len(x))
        b = np.polyfit(t, x, 1)
        pred = np.polyval(b, t)
        ss_res = np.nansum((x - pred) ** 2)
        ss_tot = np.nansum((x - x.mean()) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-12)
    out['r2_trend_60'] = lp.rolling(60, min_periods=30).apply(r2_win, raw=False)
    # 2. Excess kurtosis of 20d returns (tail risk)
    out['kurt_20'] = r.rolling(20, min_periods=12).kurt()
    # 3. Lag-1 autocorrelation of returns over 20d (mean-reversion tendency)
    out['autocorr_20'] = r.rolling(20, min_periods=12).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if np.isfinite(x).all() and x.std() > 0 else np.nan, raw=True)
    # 4. Range-based volatility: mean (high-low)/close over 20d
    out['range_vol_20'] = ((h - l) / (c + 1e-12)).rolling(20, min_periods=10).mean()
    # 5. Up-day fraction over 20d (directional breadth)
    out['upday_frac_20'] = (r > 0).rolling(20, min_periods=10).mean()
    # 6. Relative momentum vs cross-sectional mean (computed later at panel level)
    # placeholder raw 20d momentum for relative construction
    out['raw_mom_20'] = ret20
    # 7. Beta vs equal-weight basket (computed later) - placeholder per-asset vol ratio
    out['vol_ratio_5x60'] = r.rolling(5, min_periods=4).std() / (vol60 + 1e-12)
    # 8. Overnight gap dispersion: std of gaps over 20d
    out['gap_vol_20'] = df['gap'].rolling(20, min_periods=10).std()
    # 9. Extreme move: |20d ret| / cross-sectional std (computed later) - placeholder
    out['abs_mom_20'] = ret20.abs()
    # 10. WTI-beta proxy: per-asset 60d corr with WTI ret (computed later) - placeholder
    out['ret60'] = ret60
    # 11. Short-term vs long-term momentum gap (reversion within trend)
    out['mom_gap_5x60'] = (ret5 - ret60 / 12.0) / (vol20 + 1e-12)
    # 12. Days since 60d low (mirror of days_since_high_60)
    roll_low = l.rolling(60, min_periods=30).min()
    out['days_since_low_60'] = (l <= roll_low).astype(float).replace(0, np.nan)
    # forward-fill day counts
    ds = out['days_since_low_60'].copy()
    count = 0
    vals = []
    for i, v in enumerate(ds.values):
        if v == 1.0:
            count = 0
            vals.append(0.0)
        elif np.isnan(v):
            count += 1
            vals.append(float(count))
        else:
            count += 1
            vals.append(float(count))
    out['days_since_low_60'] = vals
    return out

FACTORS = {s: build_factors(s) for s in SYMS}

# Panel-level constructions
idx = DATA['SPX']['date']
panel = pd.DataFrame({s: DATA[s]['close'] for s in SYMS}).set_axis(idx, axis=0)
panel = panel.reindex(idx).ffill()
pret = panel.pct_change()
ew_ret = pret.mean(axis=1)  # equal-weight basket return

# relative momentum: 20d ret minus cross-sectional mean (per date)
mom20 = pd.DataFrame({s: FACTORS[s]['raw_mom_20'] for s in SYMS}).set_axis(idx, axis=0)
cs_mean = mom20.mean(axis=1)
rel_mom = mom20.sub(cs_mean, axis=0)

# extreme move: |20d ret| / cross-sectional std of 20d ret (per date)
cs_std = mom20.std(axis=1)
mom_extreme = FACTORS['SPX']['abs_mom_20'].to_frame()  # placeholder
mom_extreme = pd.DataFrame({s: FACTORS[s]['abs_mom_20'] for s in SYMS}).set_axis(idx, axis=0)
mom_extreme = mom_extreme.div(cs_std.replace(0, np.nan), axis=0)

# beta vs equal-weight basket over 60d
def rolling_beta(x, y, win=60):
    cov = x.rolling(win, min_periods=30).cov(y)
    var = y.rolling(win, min_periods=30).var()
    return cov / (var + 1e-12)

beta_ew = pd.DataFrame({s: rolling_beta(pret[s], ew_ret) for s in SYMS}).set_axis(idx, axis=0)

# beta vs WTI over 60d
wti_r = pret['WTI']
beta_wti = pd.DataFrame({s: rolling_beta(pret[s], wti_r) for s in SYMS}).set_axis(idx, axis=0)

# assemble panel factors
FACTOR_PANELS = {}
for s in SYMS:
    base = FACTORS[s]
    FACTOR_PANELS[s] = pd.DataFrame({
        'r2_trend_60': base['r2_trend_60'],
        'kurt_20': base['kurt_20'],
        'autocorr_20': base['autocorr_20'],
        'range_vol_20': base['range_vol_20'],
        'upday_frac_20': base['upday_frac_20'],
        'rel_mom_20': rel_mom[s],
        'beta_ew_60': beta_ew[s],
        'gap_vol_20': base['gap_vol_20'],
        'mom_extreme_20': mom_extreme[s],
        'beta_wti_60': beta_wti[s],
        'mom_gap_5x60': base['mom_gap_5x60'],
        'days_since_low_60': base['days_since_low_60'],
    })

def forward_ret(sym, h):
    c = DATA[sym]['close']
    return c.shift(-h) / c - 1.0

def ic_series(factor_name, h):
    rows = []
    for s in SYMS:
        f = FACTOR_PANELS[s][factor_name].rename(s)
        fr = forward_ret(s, h).rename(s)
        rows.append(pd.concat([f, fr], axis=1))
    # align on union of dates, use asset's own calendar
    all_dates = sorted(set().union(*[r.index for r in rows]))
    fdf = pd.DataFrame(index=all_dates, columns=SYMS, dtype=float)
    rdf = pd.DataFrame(index=all_dates, columns=SYMS, dtype=float)
    for s, r in zip(SYMS, rows):
        fdf[s] = r.iloc[:, 0]
        rdf[s] = r.iloc[:, 1]
    dates, ics = [], []
    for d in fdf.index:
        f = fdf.loc[d]; r = rdf.loc[d]
        mask = f.notna() & r.notna()
        if mask.sum() < MIN_ASSETS:
            continue
        ic = f[mask].corr(r[mask], method='spearman')
        if np.isfinite(ic):
            dates.append(d); ics.append(ic)
    return pd.Series(ics, index=pd.to_datetime(dates))

def mean_rank_turnover(factor_name, step=10):
    fdf = pd.DataFrame({s: FACTOR_PANELS[s][factor_name] for s in SYMS})
    ranks = fdf.rank(axis=1, pct=True)
    chg = (ranks - ranks.shift(step)).abs()
    return float(chg.stack().mean())

# Library signal recomputation for correlation audit (key existing factors)
def library_signals():
    sigs = {}
    r = pret
    vol20 = r.rolling(20, min_periods=10).std()
    vol60 = r.rolling(60, min_periods=15).std()
    # mom20_volproxy60
    mom20 = panel / panel.shift(20) - 1
    sigs['mom20_volproxy60'] = mom20 / (vol60 + 1e-12)
    # gain_loss_20
    sigs['gain_loss_20'] = pd.DataFrame({s: FACTORS[s]['raw_mom_20'] for s in SYMS}).set_axis(idx, axis=0)
    # vol_of_vol20x60
    sigs['vol_of_vol20x60'] = vol20 / (vol60 + 1e-12)
    # spx_corr60
    spx_r = pret['SPX']
    sigs['spx_corr60'] = pd.DataFrame({s: pret[s].rolling(60, min_periods=30).corr(spx_r) for s in SYMS}).set_axis(idx, axis=0)
    # max_consec_gain_20 / max_consec_loss_20 (approx: max run of gains)
    def max_run(series, sign):
        runs = []
        cur = 0
        for v in series:
            if (v > 0) == sign and np.isfinite(v):
                cur += 1
            else:
                if cur > 0: runs.append(cur)
                cur = 0
        return max(runs) if runs else 0
    mcg = pd.DataFrame(index=idx, columns=SYMS, dtype=float)
    mcl = pd.DataFrame(index=idx, columns=SYMS, dtype=float)
    for s in SYMS:
        rr = DATA[s]['ret']
        mc_g = rr.rolling(20, min_periods=10).apply(lambda x: max_run(x, True), raw=True)
        mc_l = rr.rolling(20, min_periods=10).apply(lambda x: max_run(x, False), raw=True)
        mcg[s] = mc_g.values; mcl[s] = mc_l.values
    sigs['max_consec_gain_20'] = mcg
    sigs['max_consec_loss_20'] = mcl
    # days_since_high_60 (approx via rolling max hits)
    ds = pd.DataFrame(index=idx, columns=SYMS, dtype=float)
    for s in SYMS:
        h = DATA[s]['high']
        rh = h.rolling(60, min_periods=30).max()
        hit = (h >= rh).astype(float)
        cnt = 0; vals = []
        for v in hit.values:
            if v == 1.0: cnt = 0; vals.append(0.0)
            else: cnt += 1; vals.append(float(cnt))
        ds[s] = vals
    sigs['days_since_high_60'] = ds
    return sigs

LIB = library_signals()

def max_lib_corr(factor_name):
    fdf = pd.DataFrame({s: FACTOR_PANELS[s][factor_name] for s in SYMS})
    best, names = 0.0, []
    for fid, lsig in LIB.items():
        both = pd.concat([fdf.stack().rename('x'), lsig.stack().rename('y')], axis=1).dropna()
        if len(both) < 100:
            continue
        rho = float(both['x'].corr(both['y']))
        if abs(rho) > best:
            best = abs(rho); names = [fid]
        elif abs(rho) == best:
            names.append(fid)
    return best, names

NAMES = ['r2_trend_60','kurt_20','autocorr_20','range_vol_20','upday_frac_20',
         'rel_mom_20','beta_ew_60','gap_vol_20','mom_extreme_20','beta_wti_60',
         'mom_gap_5x60','days_since_low_60']

results = {}
for fn in NAMES:
    ic10 = ic_series(fn, 10)
    ic = float(ic10.mean())
    icir = float(ic10.mean() / ic10.std()) if ic10.std() > 0 else 0.0
    hit = float((ic10 > 0).mean()) if ic >= 0 else float((ic10 < 0).mean())
    decay = {}
    for h in HORIZONS:
        s = ic_series(fn, h)
        decay[str(h)] = round(float(s.mean()), 4)
    valid = pd.DataFrame({s: FACTOR_PANELS[s][fn] for s in SYMS}).notna().sum().sum()
    total = pd.DataFrame({s: FACTOR_PANELS[s][fn] for s in SYMS}).shape[0] * 15
    cov = valid / total
    dates_ge8 = sum(1 for d in pd.DataFrame({s: FACTOR_PANELS[s][fn] for s in SYMS}).index
                    if pd.DataFrame({s: FACTOR_PANELS[s][fn] for s in SYMS}).loc[d].notna().sum() >= MIN_ASSETS)
    cov_dates = dates_ge8 / pd.DataFrame({s: FACTOR_PANELS[s][fn] for s in SYMS}).shape[0]
    to = mean_rank_turnover(fn)
    mrho, rnames = max_lib_corr(fn)
    # regime blocks
    regime = {}
    for b0, b1 in [('2020-01-01','2021-12-31'), ('2022-01-01','2022-12-31'),
                   ('2023-01-01','2024-12-31'), ('2025-01-01','2026-12-31'),
                   ('2027-01-01','2030-01-23')]:
        sub = ic10[(ic10.index >= b0) & (ic10.index <= b1)]
        if len(sub) >= 30:
            regime[b0[:4]] = {'ic': round(float(sub.mean()),4),
                              'icir': round(float(sub.mean()/sub.std()),4) if sub.std() > 0 else 0.0,
                              'n': int(len(sub))}
    last250 = ic10.tail(250)
    regime['last250'] = {'ic': round(float(last250.mean()),4),
                         'icir': round(float(last250.mean()/last250.std()),4) if last250.std() > 0 else 0.0,
                         'n': int(len(last250))}
    results[fn] = {'ic': ic, 'icir': icir, 'hit': hit, 'n_ic_dates': int(len(ic10)),
                   'coverage': cov, 'dates_ge8_frac': cov_dates, 'turnover_10d_rank': to,
                   'decay': decay, 'max_lib_corr': mrho, 'rho_names': rnames, 'regime': regime}
    print(f'==== {fn} ====')
    print(f'  n_ic_dates={len(ic10)} ic={ic:.4f} icir={icir:.4f} hit={hit:.3f}')
    print(f'  coverage_asset_days={cov:.3f} dates_ge8={cov_dates:.3f} turnover={to:.3f}')
    print(f'  decay={decay}')
    print(f'  max_lib_corr={mrho:.4f} vs {rnames}')
    print(f'  regime={json.dumps(regime)}')

with open('scripts/miner3_20300124_batch_results.json', 'w') as f:
    json.dump(results, f, indent=1, default=str)
print('\nSaved results to scripts/miner3_20300124_batch_results.json')
