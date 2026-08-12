"""Screener cycle 2028-02-24: regime assessment + recent factor IC evidence."""
import json
import numpy as np
import pandas as pd

ASSETS = ['000300.SH','000688.SH','SPX','HSI','N225','SX5E','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = 'date' if 'date' in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    if 'close' in df.columns:
        return df['close']
    return df.iloc[:, 0]

px = pd.DataFrame({a: load(f'../persistent/stock_data/{a}.csv') for a in ASSETS})
dxy = load('../persistent/index_data/DXY.csv').reindex(px.index).ffill()
vix = load('../persistent/index_data/VIX.csv').reindex(px.index).ffill()
px = px.dropna(how='all')
print('panel dates:', px.index.min().date(), '->', px.index.max().date(), 'shape', px.shape)

ret = px.pct_change()
close = px

# ---------------- Regime ----------------
ma60 = close.rolling(60).mean()
above_ma60 = (close > ma60).iloc[-1]
trend20 = close.iloc[-1] / close.shift(20).iloc[-1] - 1
r20_avg = trend20.mean()
trend_strength = abs(r20_avg) / (trend20.std() + 1e-9)
rv20 = ret.tail(20).std() * np.sqrt(252)
avg_rv20 = rv20.mean()
c = ret.tail(20).corr()
mask = np.triu(np.ones(c.shape, dtype=bool), 1)
avg_corr = c.values[mask].mean()
vix_level = vix.iloc[-1]
vix_pctile = (vix.iloc[-1] < vix).mean()
dxy_trend20 = dxy.iloc[-1] / dxy.shift(20).iloc[-1] - 1
dd = (close / close.rolling(120, min_periods=30).max() - 1).tail(20).min().mean()

print('\n=== REGIME (as of', px.index[-1].date(), ') ===')
print('20d avg return across 15 assets: %.4f' % r20_avg)
print('fraction above MA60: %.2f' % above_ma60.mean())
print('trend strength (abs avg / disp): %.3f' % trend_strength)
print('avg 20d realized vol (ann): %.3f' % avg_rv20)
print('avg pairwise corr 20d: %.3f' % avg_corr)
print('VIX: %.1f (percentile %.2f)' % (vix_level, vix_pctile))
print('DXY 20d trend: %.4f' % dxy_trend20)
print('avg maxDD (20d window over 120d): %.4f' % dd)
print('\nper-asset 20d trend / aboveMA60 / rv20:')
for a in ASSETS:
    print('  %-10s r20=%7.3f  aboveMA60=%s  rv20=%.3f' % (a, trend20[a], above_ma60[a], rv20[a]))

# ---------------- Factor panels (dict name -> DF dates x assets) ----------------
FV = {}

FV['mom_10d_skip5'] = close.shift(5) / close.shift(15) - 1
FV['mom_120d_skip5'] = close.shift(5) / close.shift(125) - 1

def trend_r2_panel(pxdf, w=30):
    t = np.arange(w)
    lc = np.log(pxdf)
    out = pd.DataFrame(np.nan, index=pxdf.index, columns=pxdf.columns)
    arr_lc = lc.values
    for i in range(w - 1, len(pxdf)):
        y = arr_lc[i - w + 1: i + 1, :]
        valid = ~np.isnan(y).any(axis=0)
        if not valid.any():
            continue
        yv = y[:, valid]
        cov = np.cov(yv, t, rowvar=False)[-1, :-1]
        var_t = np.var(t)
        var_y = np.var(yv, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            r2 = np.sign(cov) * cov ** 2 / (var_t * var_y)
        out.iloc[i, valid] = np.where(var_y > 0, r2, np.nan)
    return out
FV['trend_r2_30_signed'] = trend_r2_panel(close, 30)

pos2 = ret.clip(lower=0) ** 2
neg2 = ret.clip(upper=0) ** 2
FV['semi_down_ratio_20'] = np.sqrt(neg2.rolling(20).mean()) / np.sqrt(pos2.rolling(20).mean()) - 1

def tuw_panel(pxdf, w=120):
    roll_max = pxdf.rolling(w, min_periods=30).max()
    day_num = np.arange(len(pxdf))
    out = pd.DataFrame(np.nan, index=pxdf.index, columns=pxdf.columns)
    last_reset = np.full(pxdf.shape[1], day_num[0], dtype=float)
    vals = pxdf.values
    rm = roll_max.values
    for i in range(len(pxdf)):
        above = vals[i] >= rm[i] - 1e-12
        last_reset[above] = day_num[i]
        out.iloc[i] = day_num[i] - last_reset
    return out
FV['time_under_water_120'] = tuw_panel(close, 120)

FV['kurt_20'] = ret.rolling(20, min_periods=8).kurt()
FV['vol_of_vol20x60'] = ret.rolling(20).std().rolling(60).std()
FV['tail_ratio_20'] = ret.rolling(20, min_periods=10).quantile(0.95) / ret.rolling(20, min_periods=10).quantile(0.05).abs()

def roll_beta_panel(x, y, w=60):
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    ax = x.values
    ay = np.asarray(y.values, dtype=float)
    for i in range(w - 1, len(x)):
        a = ax[i - w + 1: i + 1, :]
        b = ay[i - w + 1: i + 1]
        va = np.var(a, axis=0)
        valid = ~np.isnan(a).any(axis=0) & ~np.isnan(b) & (va > 1e-12)
        if not valid.any():
            continue
        am = a[:, valid] - a[:, valid].mean(axis=0)
        bm = b - b.mean()
        cov = (am * bm[:, None]).mean(axis=0)
        out.iloc[i, valid] = cov / va[valid]
    return out

FV['WTI_BETA_60'] = roll_beta_panel(ret, px['WTI'].pct_change(), 60)
FV['dxy_beta_60'] = roll_beta_panel(ret, dxy.pct_change(), 60)
FV['vix_beta_cond_60x20'] = -roll_beta_panel(ret, vix.pct_change(), 60) * (vix / vix.shift(20) - 1).values[:, None]

factors = list(FV.keys())
print('\n=== FACTOR RECENT IC (rank IC vs 10d fwd ret) ===')
fwd = px.shift(-10) / px - 1

def rank_ic_series(fdf, fwd_ret, dates):
    ics = []
    for d in dates:
        x = fdf.loc[d]
        y = fwd_ret.loc[d]
        m = x.notna() & y.notna()
        if m.sum() < 8:
            continue
        xx = x[m].rank()
        yy = y[m].rank()
        if xx.std() == 0 or yy.std() == 0:
            continue
        ics.append(np.corrcoef(xx, yy)[0, 1])
    return np.array(ics)

for win, label in [(120, 'recent120'), (250, 'full250')]:
    dates = px.index[-win - 10:-10]
    print(f'--- {label} ---')
    for f in factors:
        ics = rank_ic_series(FV[f], fwd, dates)
        if len(ics) < 30:
            print(f'  {f:24s} n={len(ics)} too few')
            continue
        ic = ics.mean()
        icir = ic / (ics.std() + 1e-9) * np.sqrt(len(ics))
        hit = (ics > 0).mean()
        print(f'  {f:24s} IC={ic:+.4f} ICIR={icir:+.2f} hit={hit:.2f} n={len(ics)}')

import pickle
with open('scripts/screener_fv_20280224.pkl', 'wb') as fh:
    pickle.dump({'FV': FV, 'px': px}, fh)
print('\nsaved factor panels.')
