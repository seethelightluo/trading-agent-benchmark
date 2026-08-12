"""miner_1 factor exploration batch - 2029-09-06 cycle.
Validate NEW candidate factors (avoiding evicted ideas: carry*, cn10y_corr, cryptobeta_cond,
downside_freq, drawup, eff_ratio, hl_rank, max_gain, mom_curve_volscale, orth_mom20,
range_pos_120d, ret_skew_10, sharpe_20, updown_vol_ratio, vol_price_corr, vol_surge).

Data restricted to <= visible_through (2029-09-05). No lookahead.
"""
import numpy as np
import pandas as pd

VISIBLE = '2029-09-05'
SYMS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load(sym):
    df = pd.read_csv(f'../persistent/stock_data/{sym}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= VISIBLE].reset_index(drop=True)
    for c in ['open','close','high','low','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['ret'] = df['close'].pct_change()
    return df

DATA = {s: load(s) for s in SYMS}

def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / (dn + 1e-12)
    return 100 - 100/(1+rs)

def max_drawdown_60(close):
    roll_max = close.rolling(60, min_periods=30).max()
    return close / roll_max - 1.0

def build_factors(sym):
    df = DATA[sym]
    c = df['close']; o = df['open']; h = df['high']; l = df['low']; v = df['volume']; r = df['ret']
    vol20 = r.rolling(20, min_periods=10).std()
    vol60 = r.rolling(60, min_periods=15).std()
    vol5 = r.rolling(5, min_periods=4).std()
    ret5 = c / c.shift(5) - 1
    ret20 = c / c.shift(20) - 1
    out = pd.DataFrame(index=df['date'])
    # 1. short-term vol-scaled reversal
    out['rev_5d_vol20'] = -ret5 / (vol20 + 1e-12)
    # 2. 20d move z-scored by 60d vol (exhaustion)
    out['exhaust_20x60'] = ret20 / (vol60 * np.sqrt(20) + 1e-12)
    # 3. recent acceleration: 5d ret minus avg 5d pace of 20d ret, vol-scaled
    out['accel_5x20'] = (ret5 - 0.25 * ret20) / (vol20 + 1e-12)
    # 4. overnight gap drift
    out['overnight_20'] = (o / c.shift(1) - 1).rolling(20, min_periods=10).mean()
    # 5. RSI14
    out['rsi_14'] = rsi(c, 14)
    # 6. Bollinger bandwidth 20d
    mid = c.rolling(20, min_periods=10).mean()
    sd = c.rolling(20, min_periods=10).std()
    out['bb_bandwidth_20'] = (4 * sd) / (mid + 1e-12)
    # 7. vol trend 20 vs 60
    out['vol_trend_20x60'] = vol20 / (vol60 + 1e-12)
    # 8. vol trend 5 vs 20
    out['vol_trend_5x20'] = vol5 / (vol20 + 1e-12)
    # 9. correlation regime shift vs SPX
    spx_r = DATA['SPX']['ret']
    corr20 = r.rolling(20, min_periods=12).corr(spx_r)
    corr60 = r.rolling(60, min_periods=30).corr(spx_r)
    out['corr_chg_spx_20x60'] = corr20 - corr60
    # 10. 60d skewness
    out['skew_60'] = r.rolling(60, min_periods=30).skew()
    # 11. Amihud illiquidity 20d (volume may be zero for some assets -> NaN)
    out['amihud_20'] = (r.abs() / (v + 1e-9)).rolling(20, min_periods=10).mean()
    # 12. up-beta minus down-beta vs SPX (asymmetry)
    spx_pos = (spx_r > 0)
    spx_neg = (spx_r < 0)
    up_beta = r.where(spx_pos).rolling(60, min_periods=15).cov(spx_r.where(spx_pos)) / spx_r.where(spx_pos).rolling(60, min_periods=15).var()
    dn_beta = r.where(spx_neg).rolling(60, min_periods=15).cov(spx_r.where(spx_neg)) / spx_r.where(spx_neg).rolling(60, min_periods=15).var()
    out['beta_asym_spx_60'] = up_beta - dn_beta
    # 13. 60d max drawdown depth
    out['maxdd_60'] = max_drawdown_60(c)
    # 14. close z-score vs 60d mean
    mu60 = c.rolling(60, min_periods=30).mean()
    sd60 = c.rolling(60, min_periods=30).std()
    out['zscore_close_60'] = (c - mu60) / (sd60 + 1e-12)
    return out

FACTORS = {s: build_factors(s) for s in SYMS}

def forward_ret(sym, h):
    c = DATA[sym]['close']
    return c.shift(-h) / c - 1.0

def ic_series(factor_name, h):
    """Daily cross-sectional Spearman IC (>=8 valid obs)."""
    rows = []
    for s in SYMS:
        f = FACTORS[s][factor_name]
        fr = forward_ret(s, h)
        df = pd.DataFrame({'f': f, 'r': fr}).dropna()
        rows.append(df.assign(sym=s))
    allx = pd.concat(rows)
    ics = []
    for dt, grp in allx.groupby(level=0) if False else allx.groupby(allx.index):
        if grp['f'].nunique() < 2 or grp['r'].nunique() < 2:
            continue
        if len(grp) < 8:
            continue
        ic = grp['f'].rank().corr(grp['r'].rank())
        if np.isfinite(ic):
            ics.append((dt, ic))
    if not ics:
        return pd.Series(dtype=float)
    s = pd.Series([x[1] for x in ics], index=pd.DatetimeIndex([x[0] for x in ics]))
    return s.sort_index()

def summarize(name, h=10):
    s = ic_series(name, h)
    if len(s) < 30:
        return None
    ic = s.mean()
    icir = s.mean() / (s.std() + 1e-12)
    hit = (s > 0).mean()
    return {'ic': ic, 'icir': icir, 'hit': hit, 'n': len(s), 'last250_ic': s.tail(250).mean(),
            'last250_icir': s.tail(250).mean() / (s.tail(250).std() + 1e-12)}

# coverage per factor
cov = {}
for name in FACTORS['SPX'].columns:
    n_valid = 0
    for s in SYMS:
        n_valid += int(FACTORS[s][name].notna().sum())
    cov[name] = n_valid

print('=== CANDIDATE FACTORS @ horizon 10 (window 2020-01..2029-09-05) ===')
print(f"{'factor':22s} {'IC':>8s} {'ICIR':>8s} {'hit':>6s} {'n_dates':>7s} {'last250IC':>9s} {'last250ICIR':>10s} {'validobs':>8s} {'GATE':>5s}")
results = {}
for name in FACTORS['SPX'].columns:
    r = summarize(name, 10)
    if r is None:
        print(f"{name:22s} insufficient IC obs")
        continue
    results[name] = r
    gate = 'PASS' if (abs(r['ic']) >= 0.007 and abs(r['icir']) >= 0.084) else ''
    print(f"{name:22s} {r['ic']:8.4f} {r['icir']:8.4f} {r['hit']:6.3f} {r['n']:7d} {r['last250_ic']:9.4f} {r['last250_icir']:10.4f} {cov[name]:8d} {gate:>5s}")

print()
print('=== DECAY (IC by horizon) for gate-passing candidates ===')
for name, r in results.items():
    if abs(r['ic']) >= 0.007 and abs(r['icir']) >= 0.084:
        dec = {}
        for h in [1, 2, 3, 5, 10, 20]:
            sh = ic_series(name, h)
            dec[h] = round(float(sh.mean()), 4) if len(sh) > 30 else None
        print(name, dec)
