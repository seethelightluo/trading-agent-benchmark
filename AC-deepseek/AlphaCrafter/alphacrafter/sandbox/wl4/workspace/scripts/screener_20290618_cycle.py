"""Screener cycle 2029-06-18: regime assessment + recent live IC for active factors.
Data visible window: <= 2029-06-15 (last completed trading day before decision date).
No backtest/step usage; pure factor analytics on the 15-asset cross-section.
"""
import pandas as pd, numpy as np, glob, os, json

CUT = '2029-06-15'

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

print('\n=== regime ===')
for w in [20, 60, 120, 252]:
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

print('\n=== asset snapshot (60d ret, 20d vol ann) ===')
r60 = px.iloc[-1] / px.iloc[-61] - 1
vol20 = rets.iloc[-20:].std() * np.sqrt(252)
snap = pd.DataFrame({'r60': r60, 'vol20': vol20}).round(4)
print(snap.to_string())

# MA trend of equal-weight market
ma20 = mkt.rolling(20).mean()
ma60 = mkt.rolling(60).mean()
ma200 = mkt.rolling(200).mean()
print(f'\nmkt vs MA20/MA60/MA200: {mkt.iloc[-1]*100:.2f} / {ma20.iloc[-1]*100:.2f} / {ma60.iloc[-1]*100:.2f} / {ma200.iloc[-1]*100:.2f}')

# individual asset MA position
print('\n=== asset MA position (last vs MA50/MA200) ===')
for a in live:
    c = px[a]
    ma50 = c.rolling(50).mean().iloc[-1]
    ma200 = c.rolling(200).mean().iloc[-1]
    last = c.iloc[-1]
    print(f'{a:10s} last={last:10.2f} vs MA50 {100*(last/ma50-1):+6.1f}%  vs MA200 {100*(last/ma200-1):+6.1f}%')

# ---- factor computation (match strategy.py) ----
print('\n=== factor recent IC (h=10 rank IC, direction-adjusted) ===')

def rolling_beta_series(y, x, win=60, min_obs=40):
    z = pd.concat([y.rename('y'), x.rename('x')], axis=1).dropna()
    out = pd.Series(index=y.index, dtype=float)
    for i in range(len(z)):
        if i < win:
            continue
        seg = z.iloc[i-win:i]
        if len(seg) < min_obs:
            continue
        var = float(seg.x.var())
        if var <= 1e-14:
            continue
        out.iloc[i] = float(seg.y.cov(seg.x) / var)
    return out

dnmkt = mkt.clip(upper=0.0)
cn10y_ret = px['CN10Y'].pct_change()

f_rates = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
f_dn = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
f_mom = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
for a in live:
    y = px[a].pct_change()
    f_rates[a] = rolling_beta_series(y, cn10y_ret, 60, 40)
    f_dn[a] = rolling_beta_series(y, dnmkt, 60, 40)
    r = px[a].pct_change()
    mom_fast = px[a] / px[a].shift(20) - 1.0
    mom_slow = px[a] / px[a].shift(60) - 1.0
    vol = r.rolling(20).std()
    f_mom[a] = (mom_fast - mom_slow) / vol

fwd = px.shift(-10) / px - 1.0

def rank_ic_series(fval, start, end, min_valid=6):
    fv = fval.loc[start:end]
    fr = fwd.loc[start:end]
    ics = []
    for d in fv.index:
        x = fv.loc[d]; y = fr.loc[d]
        m = x.notna() & y.notna()
        if m.sum() < min_valid:
            continue
        ics.append((d, x[m].rank().corr(y[m].rank())))
    if not ics:
        return pd.Series(dtype=float)
    return pd.Series(dict(ics))

fac_defs = {
    'rate_beta_cn10y_60d': (f_rates, -1),
    'dn_mkt_beta_60d': (f_dn, +1),
    'vol_adj_mom_accel_20x60': (f_mom, +1),
}

for name, (fv, direction) in fac_defs.items():
    print(f'\n--- {name} (dir {direction:+d}) ---')
    for label, win in [('90d', 90), ('180d', 180), ('252d', 252)]:
        start = px.index[-win]
        s = rank_ic_series(fv, start, CUT)
        if len(s) < 10:
            print(f'  {label}: n={len(s)} insufficient')
            continue
        ic = s.mean() * direction
        icir = ic / (s.std() / np.sqrt(len(s))) if s.std() > 0 else np.nan
        hit = (s * direction > 0).mean()
        print(f'  {label}: n={len(s):3d} adj_IC={ic:+.4f} ICIR={icir:+.3f} hit={hit*100:.0f}%')
    # long-run
    s = rank_ic_series(fv, px.index[-756], CUT)
    if len(s) >= 50:
        ic = s.mean() * direction
        icir = ic / (s.std() / np.sqrt(len(s))) if s.std() > 0 else np.nan
        print(f'  756d: n={len(s):3d} adj_IC={ic:+.4f} ICIR={icir:+.3f}')

# ---- pairwise factor IC correlation (252d) ----
print('\n=== pairwise factor IC correlation (252d) ===')
names = list(fac_defs.keys())
ic_series = {}
for name, (fv, direction) in fac_defs.items():
    s = rank_ic_series(fv, px.index[-252], CUT)
    ic_series[name] = s
import itertools
for a, b in itertools.combinations(names, 2):
    sa, sb = ic_series[a].align(ic_series[b], join='inner')
    if len(sa) >= 30:
        print(f'  {a} vs {b}: corr={sa.corr(sb):+.3f} (n={len(sa)})')

# ---- current signal snapshot (last visible date) ----
print('\n=== current factor signal snapshot (last visible date = %s) ===' % px.index[-1].date())
last_date = px.index[-1]
snap_out = {}
for name, (fv, direction) in fac_defs.items():
    row = fv.loc[last_date].dropna()
    # rank 0..1 (neutral = 0.5 for stale)
    r = row.rank(pct=True)
    snap_out[name] = r
    print(f'{name}:')
    for a in r.sort_values(ascending=False).index:
        print(f'    {a:10s} val={row[a]:+.4f} rank={r[a]:.2f}')

# composite z-score direction
print('\n=== composite ensemble tilt (weighted rank, dir-adjusted) ===')
ens = {'rate_beta_cn10y_60d': (0.45, -1), 'vol_adj_mom_accel_20x60': (0.35, +1), 'dn_mkt_beta_60d': (0.20, +1)}
comp = pd.Series(dtype=float)
for name, (w, d) in ens.items():
    comp = comp.add(snap_out[name] * d * w, fill_value=0.0)
# stale assets neutral (0.5 * sum weights) -> center
comp = comp - 0.5 * sum(w for w, _ in ens.values())
for a in comp.sort_values(ascending=False).index:
    print(f'    {a:10s} composite={comp[a]:+.4f}')
