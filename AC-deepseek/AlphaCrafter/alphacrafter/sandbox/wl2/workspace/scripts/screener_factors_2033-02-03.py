"""Compute live cross-sectional values of candidate factors through 2033-02-02 (no leakage)."""
import pandas as pd, numpy as np

END = '2033-02-02'
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load(a):
    df = pd.read_csv(f'../persistent/stock_data/{a}.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df[df['date'] <= END].reset_index(drop=True)

data = {a: load(a) for a in ASSETS}
closes = pd.DataFrame({a: data[a].set_index('date')['close'] for a in ASSETS}).sort_index()
rets = closes.pct_change()

def rankz(s):
    return s.rank(pct=True)

out = {}
# 1) max_consec_gain_20
mg = {}
for a in ASSETS:
    r = rets[a].dropna().tail(20).values
    c = m = 0
    for x in r:
        c = c + 1 if x > 0 else 0
        m = max(m, c)
    mg[a] = m
out['max_consec_gain_20'] = mg
# 2) mom_180d_skip5
out['mom_180d_skip5'] = {a: (closes[a].iloc[-6]/closes[a].iloc[-1] - 1) if len(closes[a].dropna()) > 180 else np.nan for a in ASSETS}
# 3) downbeta_spx_60
db = {}
spx = rets['SPX']
for a in ASSETS:
    rr = pd.concat([rets[a], spx], axis=1, keys=['a','m']).dropna().tail(60)
    sub = rr[rr['m'] < 0]
    db[a] = np.polyfit(sub['m'], sub['a'], 1)[0] if len(sub) >= 15 else np.nan
out['downbeta_spx_60'] = db
# 4) spx_corr60
out['spx_corr60'] = {a: rets[a].corr(spx) for a in ASSETS}
# 5) range_pos_252
rp = {}
for a in ASSETS:
    s = closes[a].dropna().tail(252)
    if len(s) >= 252 and s.max() != s.min():
        rp[a] = (s.iloc[-1] - s.min()) / (s.max() - s.min())
    else:
        rp[a] = np.nan
out['range_pos_252'] = rp
# extras for regime read
out['mom_20d_skip5'] = {a: (closes[a].iloc[-6]/closes[a].iloc[-1] - 1) for a in ASSETS}
out['gain_loss_20'] = {}
for a in ASSETS:
    r = rets[a].dropna().tail(20)
    out['gain_loss_20'][a] = r[r>0].sum() / abs(r[r<0].sum()) if (r[r<0].sum()!=0) else np.nan
out['calmness_20'] = {}
for a in ASSETS:
    r = rets[a].dropna().tail(20)
    out['calmness_20'][a] = (abs(r) < 0.5*r.std()).mean() if r.std()>0 else np.nan
out['days_since_high_60'] = {}
for a in ASSETS:
    s = closes[a].dropna().tail(60)
    hi = s.max(); idx_hi = s[s==hi].index[-1]
    out['days_since_high_60'][a] = (s.index[-1]-idx_hi).days

print(f"{'asset':10s}", end='')
for k in out: print(f"{k:>20s}", end='')
print()
for a in ASSETS:
    print(f"{a:10s}", end='')
    for k, d in out.items():
        v = d.get(a, np.nan)
        print(f"{v if isinstance(v,(int,np.integer)) else (f'{v:19.3f}' if pd.notna(v) else '                 NaN'):>20s}", end='')
    print()

print('\n=== rank (pct) ===')
for a in ASSETS:
    print(f"{a:10s}", end='')
    for k, d in out.items():
        v = d.get(a, np.nan)
        if pd.isna(v): print(f"{'NaN':>20s}", end='')
        else:
            s = pd.Series(d); r = s.rank(pct=True)[a]
            print(f"{r:20.3f}", end='')
    print()
