"""SCREENER regime & factor snapshot - data through 2033-02-02 only (NO future leakage).
Computes cross-sectional factor values for the 15-instrument universe and regime metrics."""
import pandas as pd, numpy as np, json

END = '2033-02-02'
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load(a):
    df = pd.read_csv(f'../persistent/stock_data/{a}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= END].reset_index(drop=True)
    return df

data = {a: load(a) for a in ASSETS}
closes = pd.DataFrame({a: data[a].set_index('date')['close'] for a in ASSETS}).sort_index()
rets = closes.pct_change()
dates = closes.index

# ---- regime table ----
def ret(s, n):
    return s.iloc[-1] / s.iloc[-1-n] - 1 if len(s) > n else np.nan

rows = {}
for a in ASSETS:
    s = closes[a].dropna(); r = rets[a].dropna()
    vol20 = r.tail(20).std() * np.sqrt(252)
    rng252 = (s.iloc[-1] - s.tail(252).min()) / (s.tail(252).max() - s.tail(252).min()) if len(s) >= 252 else np.nan
    # days since 60d high
    w = s.tail(60); hi = w.max(); idx_hi = w[w == hi].index[-1]
    dsh = (s.index[-1] - idx_hi).days
    # max consecutive gain days (20d): longest run of positive daily returns
    rr = r.tail(20).values
    mg = ml = 0; cg = cl_ = 0
    for x in rr:
        if x > 0: cg += 1; cl_ = 0
        elif x < 0: cl_ += 1; cg = 0
        mg = max(mg, cg); ml = max(ml, cl_)
    rows[a] = dict(r20=ret(s,20)*100, r60=ret(s,60)*100, r180=ret(s,180)*100,
                   vol20=vol20*100, rng252=rng252, dsh60=dsh, mg20=mg, ml20=ml)

tbl = pd.DataFrame(rows).T
pd.set_option('display.width', 200)
print('=== REGIME SNAPSHOT (through 2033-02-02) ===')
print(tbl.round(2).sort_values('r20', ascending=False).to_string())

# ---- cross-sectional metrics for factor read ----
spx = rets['SPX']
downmask = spx < 0
print('\n=== downside beta 60d to SPX ===')
db = {}
for a in ASSETS:
    rr = pd.concat([rets[a], spx], axis=1, keys=['a','m']).dropna().tail(60)
    sub = rr[rr['m'] < 0]
    if len(sub) >= 15:
        db[a] = np.polyfit(sub['m'], sub['a'], 1)[0]
    else:
        db[a] = np.nan
print(pd.Series(db).round(3).sort_values(ascending=False).to_string())

print('\n=== 60d correlation to SPX ===')
c60 = {a: rets[a].corr(spx) for a in ASSETS}
print(pd.Series(c60).round(3).sort_values(ascending=False).to_string())

# ---- macro observation series ----
print('\n=== MACRO (index_data) ===')
for m in ['VIX','DXY','USDJPY','EURUSD','USDCNY']:
    d = pd.read_csv(f'../persistent/index_data/{m}.csv')
    d['date'] = pd.to_datetime(d['date']); d = d[d['date'] <= END]
    c = d['close'] if 'close' in d.columns else d.iloc[:, 1]
    print(f'{m:8s} last {c.iloc[-1]:10.3f}  20d chg {c.iloc[-1]/c.iloc[-21]-1:+.2%}  60d chg {c.iloc[-1]/c.iloc[-61]-1:+.2%}')

# ---- pairwise avg correlation (dispersion regime) ----
cmat = rets.tail(60).corr()
import numpy as np
vals = cmat.values[np.triu_indices_from(cmat.values, k=1)]
print(f'\n=== mean pairwise 60d corr (15 assets): {np.nanmean(vals):.3f} ===')
