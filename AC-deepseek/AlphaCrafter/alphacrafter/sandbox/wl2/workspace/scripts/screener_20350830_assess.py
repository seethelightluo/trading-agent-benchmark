"""Screener cycle 2035-08-30: market regime assessment + factor signal/IC computation.
Uses ONLY data through visible_through=2035-08-29 (no lookahead)."""
import pandas as pd, numpy as np, json, os

VISIBLE = '2035-08-29'
SYMS = ['000300.SH','000688.SH','BTC','CN10Y','COPPER','ETH','HSI','N225','NDX','SOX','SPX','SX5E','US10Y','WTI','XAU']

closes = {}
for s in SYMS:
    df = pd.read_csv(f'../persistent/stock_data/{s}.csv')
    df = df[df['date'] <= VISIBLE].copy()
    df['date'] = pd.to_datetime(df['date'])
    closes[s] = df.set_index('date')['close']
px = pd.DataFrame(closes).sort_index()
px = px.ffill()
print('price matrix shape:', px.shape, 'last date:', px.index[-1].date())

rets = px.pct_change()

# ---------- regime metrics ----------
def r(period):
    return px.iloc[-1] / px.iloc[-1-period] - 1.0 if len(px) > period else np.nan

regime = {}
regime['last_date'] = str(px.index[-1].date())
regime['r20'] = {s: round(float(r(20)[s]), 4) for s in SYMS}
regime['r60'] = {s: round(float(r(60)[s]), 4) for s in SYMS}
regime['r252'] = {s: round(float(r(252)[s]), 4) for s in SYMS}

# range position 252
min252 = px.rolling(252, min_periods=30).min()
max252 = px.rolling(252, min_periods=30).max()
rpos = (px - min252) / (max252 - min252)
regime['rpos252'] = {s: round(float(rpos[s].iloc[-1]), 4) for s in SYMS}

# realized vol 20d annualized
vol20 = rets.rolling(20).std() * np.sqrt(252)
regime['vol20_ann'] = {s: round(float(vol20[s].iloc[-1]), 4) for s in SYMS}

# VIX from index_data
try:
    vix = pd.read_csv('../persistent/index_data/VIX.csv')
    vix['date'] = pd.to_datetime(vix['date'])
    vix = vix[vix['date'] <= VISIBLE]
    regime['vix_last'] = float(vix['close'].iloc[-1])
    regime['vix_mean60'] = float(vix['close'].tail(60).mean())
except Exception as e:
    regime['vix_last'] = None
    regime['vix_mean60'] = None

# DXY, USDJPY observation signals
for obs in ['DXY','USDJPY','USDCNY','EURUSD']:
    try:
        o = pd.read_csv(f'../persistent/index_data/{obs}.csv')
        o['date'] = pd.to_datetime(o['date'])
        o = o[o['date'] <= VISIBLE]
        regime[f'{obs}_r20'] = round(float(o['close'].iloc[-1]/o['close'].iloc[-21]-1), 4) if len(o) > 21 else None
        regime[f'{obs}_last'] = float(o['close'].iloc[-1])
    except Exception:
        pass

# ---------- factor computations (signal at t, forward 10d return) ----------
fwd = px.shift(-10) / px - 1.0  # forward 10d return

def spearman_ic(factor_series, fwd_ret):
    """cross-sectional spearman IC per date; factor_series: DataFrame dates x assets"""
    ics = []
    dates = factor_series.index.intersection(fwd_ret.index)
    for dt in dates:
        f = factor_series.loc[dt]
        r_ = fwd_ret.loc[dt]
        m = f.notna() & r_.notna()
        if m.sum() >= 8:
            ic = f[m].rank().corr(r_[m].rank())
            if pd.notna(ic):
                ics.append(ic)
    return pd.Series(ics, index=dates)

def ic_stats(ics, label):
    if len(ics) == 0:
        return {'label': label, 'n': 0}
    ic_mean = ics.mean()
    ic_std = ics.std(ddof=1)
    icir = ic_mean / ic_std if ic_std > 0 else np.nan
    return {'label': label, 'n': len(ics), 'ic': round(float(ic_mean), 5),
            'icir': round(float(icir), 4), 'hit': round(float((ics > 0).mean()), 3)}

factors = {}

# 1. max_consec_gain_20: longest run of consecutive up days within trailing 20d
up = (rets > 0).astype(int)
def max_consec(series_20d):
    # series_20d: boolean array length 20
    best = cur = 0
    for v in series_20d:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best
mc = up.rolling(20).apply(lambda x: max_consec(x.values), raw=True)
factors['max_consec_gain_20'] = mc

# 2. spx_corr60
spx_ret = rets['SPX']
corr60 = rets.rolling(60, min_periods=15).corr(spx_ret)
factors['spx_corr60'] = corr60

# 3. mom_180d_skip5
factors['mom_180d_skip5'] = px.shift(5) / px.shift(185) - 1.0

# 4. range_pos_252 (reuse rpos)
factors['range_pos_252'] = rpos

# 5. downbeta_spx_60
down_mask = (spx_ret < 0)
def downbeta(asset_ret, spx_ret, window=60, min_down=15):
    out = pd.Series(np.nan, index=asset_ret.index)
    for i in range(window, len(asset_ret)):
        a = asset_ret.iloc[i-window:i]
        s = spx_ret.iloc[i-window:i]
        m = s < 0
        if m.sum() >= min_down:
            cov = a[m].cov(s[m])
            var = s[m].var()
            if pd.notna(cov) and var > 0:
                out.iloc[i] = cov / var
    return out
db = downbeta(rets['XAU'], spx_ret)  # placeholder compute on one asset first to test
# compute for all assets
db_all = pd.DataFrame({s: downbeta(rets[s], spx_ret) for s in SYMS})
factors['downbeta_spx_60'] = db_all

print('\n=== regime ===')
print('VIX last:', regime.get('vix_last'), 'mean60:', regime.get('vix_mean60'))
print('r20:', json.dumps(regime['r20']))
print('rpos252:', json.dumps(regime['rpos252']))
print('vol20:', json.dumps(regime['vol20_ann']))

print('\n=== factor ICs (full history thru visible) ===')
full_ic = {}
for name, fser in factors.items():
    st = ic_stats(spearman_ic(fser, fwd), name)
    full_ic[name] = st
    print(st)

print('\n=== factor ICs (last 252d) ===')
recent_ic = {}
for name, fser in factors.items():
    ics = spearman_ic(fser, fwd)
    ics_recent = ics[ics.index >= pd.Timestamp('2034-08-29')] if len(ics) else ics
    st = ic_stats(ics_recent, name)
    recent_ic[name] = st
    print(st)

print('\n=== factor loads at last date ===')
for name, fser in factors.items():
    last = fser.iloc[-1]
    print(name, '->', {s: round(float(last[s]), 3) for s in SYMS})

with open('scripts/screener_20350830_regime.json', 'w') as f:
    json.dump({'regime': regime, 'full_ic': full_ic, 'recent_ic_252': recent_ic}, f, indent=1, default=str)
print('\nsaved regime+ic json')
