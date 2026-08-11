"""Screener: recompute factor values from live price data (truncated to 2027-01-13)
and evaluate cross-sectional rank IC vs forward 10d returns over recent windows."""
import numpy as np
import pandas as pd
import glob, os, json

END = '2027-01-13'          # last completed trading day visible to decisions
ASSETS = ['000300.SH','000688.SH','SPX','HSI','N225','SX5E','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
OBS = {'DXY': '../persistent/index_data/DXY.csv', 'VIX': '../persistent/index_data/VIX.csv'}

# load closes
closes = {}
for a in ASSETS:
    df = pd.read_csv(f'../persistent/stock_data/{a}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= END].set_index('date')['close']
    closes[a] = df
px = pd.DataFrame(closes).sort_index()
print('price panel', px.shape, px.index.min().date(), '->', px.index.max().date())

wti = px['WTI']
dxy = pd.read_csv(OBS['DXY']); dxy['date'] = pd.to_datetime(dxy['date']); dxy = dxy[dxy['date']<=END].set_index('date')['close'].sort_index()
vix = pd.read_csv(OBS['VIX']); vix['date'] = pd.to_datetime(vix['date']); vix = vix[vix['date']<=END].set_index('date')['close'].sort_index()

ret = px.pct_change()
wti_ret = wti.pct_change()
dxy_ret = dxy.pct_change()
vix_ret = vix.pct_change()

def roll_beta(x, y, w):
    cov = x.rolling(w).cov(y)
    var = y.rolling(w).var()
    return cov / var

def roll_kurt(s, w=20, minp=8):
    mu = s.rolling(w, min_periods=minp).mean()
    m2 = ((s - mu)**2).rolling(w, min_periods=minp).mean()
    m4 = ((s - mu)**4).rolling(w, min_periods=minp).mean()
    return m4 / m2**2 - 3.0

factors = {}
factors['mom_10d_skip5']  = px.shift(5) / px.shift(15) - 1.0
factors['mom_120d_skip5'] = px.shift(5) / px.shift(125) - 1.0

# trend_r2_30_signed: signed R2 of OLS log-price on t over 30d
logpx = np.log(px)
t_idx = np.arange(len(px))
tr2 = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
for i in range(29, len(px)):
    win = logpx.iloc[i-29:i+1]
    tt = t_idx[i-29:i+1]
    for c in px.columns:
        y = win[c].values
        if np.isfinite(y).sum() < 18: continue
        cov = np.cov(y, tt)[0,1]
        vart = np.var(tt)
        vary = np.var(y)
        if vart <= 0 or vary <= 0:
            tr2.iloc[i, tr2.columns.get_loc(c)] = 0.0
        else:
            r2 = cov**2 / (vart * vary)
            tr2.iloc[i, tr2.columns.get_loc(c)] = np.sign(cov) * r2
factors['trend_r2_30_signed'] = tr2

# semi_down_ratio_20: sqrt(mean(min(r,0)^2,20))/sqrt(mean(max(r,0)^2,20)) - 1
down = (ret.clip(upper=0)**2).rolling(20).mean().apply(np.sqrt)
up   = (ret.clip(lower=0)**2).rolling(20).mean().apply(np.sqrt)
factors['semi_down_ratio_20'] = down / up - 1.0

# time_under_water_120: days since last rolling-120d max
rmax = px.rolling(120).max()
tuw = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
for c in px.columns:
    s = px[c]; m = rmax[c]
    days = 0
    vals = []
    for i in range(len(s)):
        if i < 119:
            vals.append(np.nan); continue
        if m.iloc[i] > 0 and s.iloc[i] >= m.iloc[i] - 1e-12:
            days = 0
        else:
            days += 1
        vals.append(days)
    tuw[c] = vals
factors['time_under_water_120'] = tuw

factors['kurt_20'] = roll_kurt(ret, 20, 8)
factors['vol_of_vol20x60'] = ret.rolling(20).std().rolling(60).std()
factors['dxy_beta_60'] = roll_beta(ret, dxy_ret, 60)
factors['WTI_BETA_60'] = roll_beta(ret, wti_ret, 60)
vix_cond = -roll_beta(ret, vix_ret, 60) * (vix / vix.shift(20) - 1.0)
factors['vix_beta_cond_60x20'] = vix_cond

# forward 10d return
fwd = px.shift(-10) / px - 1.0

def rank_ic(fv, fwd_ret):
    """cross-sectional spearman IC per date"""
    out = {}
    for dt in fv.index:
        x = fv.loc[dt]; y = fwd_ret.loc[dt]
        m = x.notna() & y.notna()
        if m.sum() < 8: continue
        out[dt] = x[m].rank().corr(y[m].rank())
    return pd.Series(out)

print('\n=== Rank IC summary (vs forward 10d return) ===')
print(f"{'factor':22s} {'exp_dir':>7s} {'ic120':>7s} {'icir120':>7s} {'ic60':>7s} {'icir60':>7s} {'ic30':>7s} {'n120':>5s}")
results = {}
for name, fv in factors.items():
    ic = rank_ic(fv, fwd)
    ic = ic.sort_index()
    rows = {}
    for lab, n in [('120', 120), ('60', 60), ('30', 30)]:
        s = ic.tail(n)
        m = s.mean(); sd = s.std(ddof=1)
        rows[lab] = (m, m/sd if sd > 0 else np.nan, len(s))
    results[name] = rows
    print(f"{name:22s} {str(factors[name].columns.tolist() and 0):>7s}", end='')
    print(f" {rows['120'][0]:7.4f} {rows['120'][1]:7.2f} {rows['60'][0]:7.4f} {rows['60'][1]:7.2f} {rows['30'][0]:7.4f} {int(rows['120'][2]):5d}")

# also save IC series for inspection
import pickle
with open('scripts/_ic_series.pkl','wb') as fh:
    pickle.dump({k: rank_ic(v, fwd) for k, v in factors.items()}, fh)
print('\nsaved IC series')
