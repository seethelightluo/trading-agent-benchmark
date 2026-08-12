"""Screener cycle 2029-01-01: regime assessment + recent live IC for active factors.
Data visible window: <= 2028-12-29 (last completed trading day before decision date).
No backtest/step usage; pure factor analytics on the 15-asset cross-section.
"""
import pandas as pd, numpy as np, glob, os, json

CUT = '2028-12-29'

# ---- load 15 tradable assets ----
files = sorted(glob.glob('../persistent/stock_data/*.csv'))
px = {}
for f in files:
    sym = os.path.basename(f).replace('.csv', '')
    df = pd.read_csv(f)
    df.columns = [c.strip().lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    px[sym] = df['close'].astype(float)
px = pd.DataFrame(px).sort_index()
px = px[px.index <= CUT]

print('visible data range:', px.index.min().date(), '->', px.index.max().date(), '| rows', len(px))

# stale / frozen detection: identical closes in last 5 days
def is_stale(c):
    tail = c.iloc[-6:]
    return bool((tail.diff().dropna().abs() < 1e-12).all())
stale = [c for c in px.columns if is_stale(px[c])]
print('stale/frozen:', stale)
live = [c for c in px.columns if c not in stale]

# ---- macro observation signals ----
def load_idx(sym):
    df = pd.read_csv(f'../persistent/index_data/{sym}.csv')
    df.columns = [c.strip().lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date').sort_index()['close'].astype(float)

macro = {s: load_idx(s) for s in ['DXY', 'VIX', 'EURUSD', 'USDJPY', 'USDCNY']}
for s in macro:
    macro[s] = macro[s][macro[s].index <= CUT]

# ---- regime stats ----
rets = px.pct_change()
mkt = rets[live].mean(axis=1)  # equal-weight live-asset market

for w in [20, 60, 120]:
    r = (1 + mkt).rolling(w).apply(np.prod, raw=True) - 1
    print(f'mkt(live) {w:3d}d cum return: {r.iloc[-1]*100:+.2f}%')

for w in [20, 60]:
    v = mkt.rolling(w).std() * np.sqrt(252)
    print(f'mkt(live) {w:3d}d realized vol (ann): {v.iloc[-1]*100:.1f}%')

cs_ret20 = (1 + rets[live]).rolling(20).apply(np.prod, raw=True) - 1
cs_disp = cs_ret20.std(axis=1)
print(f'cross-sectional dispersion of 20d returns (live): last={cs_disp.iloc[-1]*100:.2f}%  '
      f'3m mean={cs_disp.iloc[-65:].mean()*100:.2f}%  1y mean={cs_disp.iloc[-252:].mean()*100:.2f}%')

corr_vals = []
for d in rets[live].index[-60:]:
    c = rets[live].loc[d-60:d].corr()
    vals = c.values[np.triu_indices_from(c.values, k=1)]
    corr_vals.append(np.nanmean(vals))
print('avg pairwise corr (live, 60d): last=%.3f  mean_3m=%.3f' % (corr_vals[-1], np.mean(corr_vals[-65:])))

for s in ['VIX', 'DXY', 'USDJPY', 'EURUSD', 'USDCNY']:
    x = macro[s]
    print(f'{s}: last={x.iloc[-1]:.2f}  60d ago={x.iloc[-61]:.2f}' if len(x) > 61 else f'{s}: last={x.iloc[-1]:.2f}')

# asset 60d returns / vol
print('\n=== asset snapshot (60d ret, 20d vol ann) ===')
r60 = px.iloc[-1] / px.iloc[-61] - 1
vol20 = rets.iloc[-20:].std() * np.sqrt(252)
snap = pd.DataFrame({'r60': r60, 'vol20': vol20}).round(4)
print(snap.to_string())

# ---- factor computation ----
def rolling_beta(y, x, win=60, min_obs=40):
    out = pd.Series(index=y.index, dtype=float)
    for i in range(len(y)):
        if i < win:
            continue
        ys, xs = y.iloc[i-win:i], x.iloc[i-win:i]
        m = ys.notna() & xs.notna()
        if m.sum() < min_obs:
            continue
        out.iloc[i] = np.polyfit(xs[m], ys[m], 1)[0]
    return out

dnmkt = mkt.where(mkt < 0, 0.0)
f_dn = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
f_cn = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
f_mom = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
for c in px.columns:
    f_dn[c] = rolling_beta(rets[c], dnmkt, 60, 40)
    f_cn[c] = rolling_beta(rets[c], px['CN10Y'].pct_change(), 60, 40)
    r20 = px[c] / px[c].shift(20) - 1
    r60 = px[c] / px[c].shift(60) - 1
    v20 = rets[c].rolling(20).std()
    f_mom[c] = (r20 - r60) / v20

facs = {'vol_adj_mom_accel_20x60': f_mom, 'dn_mkt_beta_60d': f_dn, 'rate_beta_cn10y_60d': f_cn}

fwd = px.shift(-10) / px - 1

def rank_ic(fval, fwd10, start, end, min_valid=6, assets=None):
    fv = fval.loc[start:end]
    fr = fwd10.loc[start:end]
    ics = []
    for d in fv.index:
        x = fv.loc[d]
        y = fr.loc[d]
        if assets is not None:
            x = x[assets]
            y = y[assets]
        m = x.notna() & y.notna()
        if m.sum() < min_valid:
            continue
        ics.append((d, x[m].rank().corr(y[m].rank())))
    if not ics:
        return None
    return pd.Series(dict(ics))

print('\n=== recent rank IC (h=10), LIVE assets only ===')
windows = [('2028 YTD', '2028-01-01', CUT),
           ('last 6m', '2028-06-01', CUT),
           ('last 3m', '2028-09-01', CUT),
           ('last 90td', None, None),
           ('last 60td', None, None),
           ('last 40td', None, None)]
for label, a, b in windows:
    if a is None:
        a = px.index[-91]
        b = px.index[-1]
    print(f'--- {label} ({a.date()}..{b.date()}) ---')
    for name, fv in facs.items():
        s = rank_ic(fv, fwd, a, b, assets=live)
        if s is None or len(s) == 0:
            print(f'  {name}: no data')
            continue
        ic = s.mean()
        icir = s.mean() / s.std() if s.std() > 0 else np.nan
        hit = (np.sign(s) > 0).mean()
        print(f'  {name}: n={len(s):3d} IC={ic:+.4f} ICIR={icir:+.3f} hit={hit:.2f}')

# ---- factor correlation (last 90td, live) ----
print('\n=== factor exposure correlation (last 90td, live) ===')
fdf = pd.DataFrame({k: fv[live].mean(axis=1) for k, fv in facs.items()}).iloc[-90:]
print(fdf.corr().round(3).to_string())

# ---- latest factor cross-section ----
print('\n=== latest factor exposures (2028-12-29, live) ===')
last = px.index[-1]
for name, fv in facs.items():
    x = fv.loc[last].dropna().sort_values()
    print(f'{name}:')
    print('   low :', x.head(6).round(3).to_dict())
    print('   high:', x.tail(6).round(3).to_dict())

# ---- decay: IC by horizon (last 6m) ----
print('\n=== IC by horizon (last 6m, live) ===')
for h in [1, 2, 3, 5, 10, 20]:
    fw = px.shift(-h) / px - 1
    row = []
    for name, fv in facs.items():
        s = rank_ic(fv, fw, '2028-06-01', CUT, assets=live)
        row.append(f'{name}={s.mean():+.4f}' if s is not None and len(s) else f'{name}=na')
    print(f'h={h:2d}: ' + '  '.join(row))
