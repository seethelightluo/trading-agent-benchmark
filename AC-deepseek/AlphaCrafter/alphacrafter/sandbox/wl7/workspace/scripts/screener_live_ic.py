import pandas as pd, numpy as np, json

cutoff = '2027-02-15'
assets = ['000300.SH','BTC','CN10Y','COPPER','ETH','HSI','N225','SPX','SX5E','US10Y','WTI','XAU']
closes = {}
for a in assets:
    df = pd.read_csv(f'../persistent/stock_data/{a}.csv')
    df.columns = [c.strip() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= cutoff].set_index('date')['close']
    closes[a] = df
C = pd.DataFrame(closes).dropna()
R = C.pct_change()
ew = R.mean(axis=1)

def macro(sym):
    df = pd.read_csv(f'../persistent/index_data/{sym}.csv')
    df.columns = [c.strip() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= cutoff].set_index('date')['close']
    return df

dxy = macro('DXY'); eurusd = macro('EURUSD')
rdxy = dxy.pct_change(); reurusd = eurusd.pct_change()

S = {}
# rel_mom_20d_skip5: cross-sectionally demeaned 20d momentum computed on data up to t-5
m20 = C / C.shift(20) - 1
m20s = m20.shift(5)
S['rel_mom_20d_skip5'] = m20s.sub(m20s.median(axis=1), axis=0)
# beta_ew_60d
cov = R.rolling(60).cov(ew)
var = ew.rolling(60).var()
S['beta_ew_60d'] = cov.div(var, axis=0)
# downside_vol_ratio_20 (flipped)
neg = R.clip(upper=0)
S['downside_vol_ratio_20'] = -(neg.rolling(20).std() / R.rolling(20).std())
# max_ret_20d
S['max_ret_20d'] = R.rolling(20).max()
# kurt_20d_skip5
S['kurt_20d_skip5'] = R.rolling(20).kurt().shift(5)
# eurusd_beta_cond_60x20
cov2 = R.rolling(60).cov(reurusd)
var2 = reurusd.rolling(60).var()
eb = cov2.div(var2, axis=0)
S['eurusd_beta_cond_60x20'] = eb * (eurusd / eurusd.shift(20) - 1)
# dxy_beta_cond_60x20
cov3 = R.rolling(60).cov(rdxy)
var3 = rdxy.rolling(60).var()
db = cov3.div(var3, axis=0)
S['dxy_beta_cond_60x20'] = -db * (dxy / dxy.shift(20) - 1)
# corr_ew_60: mean absolute pairwise corr with other assets
corr_out = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
rc = R.rolling(60).corr()
for a in assets:
    col = rc.xs(a, level=1) if isinstance(rc.index, pd.MultiIndex) else rc[a]
    others = [x for x in assets if x != a]
    corr_out[a] = col[others].abs().mean(axis=1)
S['corr_ew_60'] = corr_out

common = R.index
S = {k: v.reindex(common) for k, v in S.items()}

# pairwise signal correlation (stacked, last 120d)
names = list(S.keys())
flat = pd.DataFrame({k: S[k].stack() for k in names}).dropna()
corr = flat.corr()
print('=== pairwise |corr| > 0.35 (last 120d) ===')
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        c = corr.iloc[i, j]
        if abs(c) > 0.35:
            print(f'{names[i]:24s} ~ {names[j]:24s} : {c:+.3f}')

# recent IC: rank corr of signal(t) vs fwd 10d return
fwd = C.shift(-10) / C - 1
print()
print('=== live h10 IC (rank corr) ===')
for k in names:
    s = S[k]
    ics = []
    for t in common[:-10]:
        f = s.loc[t].dropna()
        r = fwd.loc[t].reindex(f.index).dropna()
        idx = f.index.intersection(r.index)
        if len(idx) >= 8:
            fr = pd.Series(f.loc[idx]).rank()
            rr = pd.Series(r.loc[idx]).rank()
            ics.append(fr.corr(rr))
    ics = pd.Series(ics)
    if len(ics):
        print(f'{k:24s} IC60={ics.tail(60).mean():+.4f} IC120={ics.tail(120).mean():+.4f} ICall={ics.mean():+.4f} n={len(ics)}')

# last 5 trading days signal snapshot (cross-sectional) for context
print()
print('=== latest signal snapshot (ranked within 12-asset alive cross-section) ===')
last = common[-1]
for k in names:
    row = S[k].loc[last]
    rk = row.rank(pct=True)
    top = rk.idxmax(); bot = rk.idxmin()
    print(f'{k:24s} top={top:10s} bot={bot:10s}')
