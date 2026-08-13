"""SCREENER analysis for cycle at 2031-07-28 (visible through 2031-07-25).

Regime assessment + recent factor IC evaluation for the 3 active factors.
Data is restricted to <= visible_through to avoid lookahead.
No backtest/step imports; analysis only.
"""
import numpy as np
import pandas as pd

VISIBLE = '2031-07-25'
WATCHLIST = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX',
             'NDX', 'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

# ---------- load ----------
frames = {}
for s in WATCHLIST:
    df = pd.read_csv(f'../persistent/stock_data/{s}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= pd.Timestamp(VISIBLE)]
    df = df.set_index('date').sort_index()
    frames[s] = df['close']
px = pd.DataFrame(frames)
px = px[~px.index.duplicated(keep='last')].sort_index()
ret = px.pct_change()

macro = {}
for m in ['VIX', 'DXY', 'USDCNY', 'USDJPY', 'EURUSD']:
    df = pd.read_csv(f'../persistent/index_data/{m}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= pd.Timestamp(VISIBLE)]
    df = df.set_index('date').sort_index()
    macro[m] = df['close']
mac = pd.DataFrame(macro)

print(f'panel: {px.index.min().date()} .. {px.index.max().date()} rows={len(px)} assets={px.shape[1]}')

# ---------- regime metrics ----------
def rets(days):
    return px.iloc[-1] / px.iloc[-1 - days] - 1.0

r20, r60, r120 = rets(20), rets(60), rets(120)
print('\n=== recent asset returns ===')
tbl = pd.DataFrame({'r20': r20, 'r60': r60, 'r120': r120}).sort_values('r60', ascending=False)
print(tbl.round(4).to_string())

ew = px.mean(axis=1)
ew_r20 = ew.iloc[-1] / ew.iloc[-21] - 1.0
ew_r60 = ew.iloc[-1] / ew.iloc[-61] - 1.0
ew_r120 = ew.iloc[-1] / ew.iloc[-121] - 1.0

above_ma60 = (px.iloc[-1] > px.rolling(60).mean().iloc[-1]).mean()
vol20 = ret.tail(20).std() * np.sqrt(252)
vol60 = ret.tail(60).std() * np.sqrt(252)

# cross-sectional dispersion: std of 20d returns across assets
disp20 = ret.tail(20).std(axis=1).mean()  # avg daily cross-sectional vol
# mean pairwise correlation of 20d returns
corr20 = ret.tail(20).corr()
np.fill_diagonal(corr20.values, np.nan)
mean_corr = corr20.stack().mean()

# MA slope: 20d change in 60d MA of equal-weight index
ma60_ew = ew.rolling(60).mean()
ma_slope = (ma60_ew.iloc[-1] / ma60_ew.iloc[-21] - 1.0)

# drawdown from 1y high
dd = px.iloc[-1] / px.rolling(252).max().iloc[-1] - 1.0

print('\n=== regime ===')
print(f'EW index r20/r60/r120: {ew_r20:.4f} / {ew_r60:.4f} / {ew_r120:.4f}')
print(f'fraction above 60d MA: {above_ma60:.2f}')
print(f'60d MA slope (20d chg): {ma_slope:.4f}')
print(f'mean realized vol 20d/60d: {vol20.mean():.4f} / {vol60.mean():.4f}')
print(f'cross-sectional daily dispersion (20d avg): {disp20:.4f}')
print(f'mean pairwise 20d return corr: {mean_corr:.3f}')
print(f'VIX last: {mac["VIX"].iloc[-1]:.2f}, 60d mean: {mac["VIX"].tail(60).mean():.2f}')
print(f'DXY last: {mac["DXY"].iloc[-1]:.2f}, 60d chg: {mac["DXY"].iloc[-1]/mac["DXY"].iloc[-61]-1:.4f}')
print(f'USDJPY last: {mac["USDJPY"].iloc[-1]:.2f}, 60d chg: {mac["USDJPY"].iloc[-1]/mac["USDJPY"].iloc[-61]-1:.4f}')
print(f'EURUSD last: {mac["EURUSD"].iloc[-1]:.2f}, 60d chg: {mac["EURUSD"].iloc[-1]/mac["EURUSD"].iloc[-61]-1:.4f}')
print(f'12m drawdowns:\n{dd.sort_values().round(4).to_string()}')

# ---------- factor signals ----------
def zscore(s):
    return (s - s.mean()) / (s.std() + 1e-12)

# 1) vol_adj_mom_accel_20x60 = (mom20 - mom60)/vol20
mom20 = px / px.shift(20) - 1.0
mom60 = px / px.shift(60) - 1.0
rv20 = ret.rolling(20).std()
f_mom = (mom20 - mom60) / rv20

# 2) dn_mkt_beta_60d: beta of asset ret on min(mkt,0), 60d window
mkt = px.mean(axis=1).pct_change()
down = mkt.where(mkt < 0, 0.0)
def rolling_beta(y, x, win=60, min_obs=40):
    out = pd.DataFrame(index=y.index, columns=y.columns, dtype=float)
    for dt in y.index:
        i = y.index.get_loc(dt)
        if i < win - 1:
            continue
        yw = y.iloc[i - win + 1:i + 1]
        xw = x.iloc[i - win + 1:i + 1]
        mask = yw.notna() & xw.notna() & np.isfinite(yw) & np.isfinite(xw)
        if mask.sum().sum() < min_obs:
            continue
        xv = xw[mask].values.ravel()
        for c in y.columns:
            yv = yw[c][mask[c]].values
            if len(yv) < min_obs:
                continue
            xm = xv[:len(yv)]
            if xm.std() == 0:
                continue
            b = np.polyfit(xm, yv, 1)[0]
            out.loc[dt, c] = b
    return out

f_beta = rolling_beta(ret, down, 60, 40)
# negative down-beta = safe haven; factor is low-downside-beta => use -beta as signal
f_beta_sig = -f_beta

# 3) rate_beta_cn10y_60d: beta of asset ret on CN10Y pct change
cn10y = px['CN10Y'].pct_change()
f_rate = rolling_beta(ret, cn10y, 60, 40)
# factor signal = beta itself (high beta = co-moves with CN rates); direction -1

# ---------- IC evaluation ----------
def fwd(panel, h=10):
    return panel.shift(-h) / panel - 1.0

def rank_ic_series(factor, fwd, min_valid=8):
    dates = factor.index.intersection(fwd.index)
    ics = {}
    for dt in dates:
        f = factor.loc[dt]; r = fwd.loc[dt]
        mask = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        if mask.sum() < min_valid:
            continue
        ic = f[mask].corr(r[mask], method='spearman')
        if np.isfinite(ic):
            ics[dt] = ic
    return pd.Series(ics, name='ic')

def summarize(ic, label):
    if len(ic) == 0:
        return {'label': label, 'n': 0}
    icm = ic.mean(); ics = ic.std(ddof=1)
    return {'label': label, 'n': len(ic), 'ic': icm,
            'icir': icm / ics if ics > 0 else np.nan,
            'hit': (ic > 0).mean(), 'ic_std': ics}

FWD10 = fwd(px, 10)
ONLINE = pd.Timestamp('2026-07-16')
RECENT = pd.Timestamp('2031-03-01')

print('\n=== factor IC (h=10) ===')
results = {}
for name, sig, direction in [('vol_adj_mom_accel_20x60', f_mom, 1),
                             ('dn_mkt_beta_60d', f_beta_sig, 1),
                             ('rate_beta_cn10y_60d', f_rate, -1)]:
    ic_all = rank_ic_series(sig, FWD10)
    ic_on = ic_all[ic_all.index >= ONLINE]
    ic_rec = ic_all[ic_all.index >= RECENT]
    s_all = summarize(ic_all, name + '_ALL')
    s_on = summarize(ic_on, name + '_ONLINE')
    s_rec = summarize(ic_rec, name + '_RECENT')
    print(f'\n{name} (direction {direction:+d})')
    for s in (s_all, s_on, s_rec):
        print(f"  {s['label']}: n={s['n']} ic={s.get('ic', float('nan')):.4f} icir={s.get('icir', float('nan')):.3f} hit={s.get('hit', float('nan')):.3f}")
    results[name] = {'all': s_all, 'online': s_on, 'recent': s_rec}

# also decay check for recent window
print('\n=== recent IC decay (h=1..20) for momentum factor ===')
for h in [1, 3, 5, 10, 20]:
    ic_h = rank_ic_series(f_mom, fwd(px, h))
    ic_h = ic_h[ic_h.index >= RECENT]
    print(f'  h={h}: ic={ic_h.mean():.4f} n={len(ic_h)}')

# ---------- quality tilt ----------
print('\n=== quality ranking (recent window, q=|ic|*|icir|) ===')
rows = []
for name, direction in [('vol_adj_mom_accel_20x60', 1), ('dn_mkt_beta_60d', 1), ('rate_beta_cn10y_60d', -1)]:
    s = results[name]['recent']
    ic = s.get('ic', np.nan); icir = s.get('icir', np.nan)
    q = abs(ic) * abs(icir) if np.isfinite(ic) and np.isfinite(icir) else 0.0
    rows.append({'factor': name, 'dir': direction, 'ic': ic, 'icir': icir,
                 'hit': s.get('hit', np.nan), 'n': s.get('n', 0), 'q': q})
rt = pd.DataFrame(rows).sort_values('q', ascending=False)
print(rt.round(4).to_string())

w = rt['q'].clip(lower=0)
if w.sum() > 0:
    w = w / w.sum()
    rt['weight'] = w
else:
    rt['weight'] = 0.0
print('\n=== proposed weights (q-tilt) ===')
print(rt.round(4).to_string())

# factor cross-correlation (recent, on common dates)
print('\n=== factor signal pairwise corr (recent common dates) ===')
sig_all = pd.concat({'vol_adj_mom_accel_20x60': f_mom, 'dn_mkt_beta_60d': f_beta_sig,
                     'rate_beta_cn10y_60d': f_rate}, axis=1)
sig_all = sig_all[sig_all.index >= RECENT]
print(sig_all.corr().round(3).to_string())
