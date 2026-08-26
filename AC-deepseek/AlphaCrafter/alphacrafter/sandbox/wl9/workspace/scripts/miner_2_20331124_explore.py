import pandas as pd, numpy as np, json
from alphacrafter.sim.utils import get_stock_daily_data

watch = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
DAYS = 2500
closes={}; highs={}; lows={}; vols={}; opens={}
for s in watch:
    df = get_stock_daily_data(symbol=s, days=DAYS)
    df['date']=pd.to_datetime(df['date']); df=df.set_index('date').sort_index()
    closes[s]=df['close']; highs[s]=df['high']; lows[s]=df['low']; vols[s]=df['volume']; opens[s]=df['open']
close_df=pd.DataFrame(closes); high_df=pd.DataFrame(highs); low_df=pd.DataFrame(lows); vol_df=pd.DataFrame(vols)
ret=close_df.pct_change()

def compute_ic(factor, horizon=10):
    fwd = close_df.pct_change(horizon).shift(-horizon)
    rows=[]
    for dt,row in factor.iterrows():
        f=row.dropna(); fw=fwd.loc[dt]
        common=f.index.intersection(fw.dropna().index)
        if len(common)>=8:
            ic=np.corrcoef(f[common],fw[common])[0,1]
            if np.isfinite(ic): rows.append((dt,ic,len(common)))
    return rows

def report(name, fac):
    ic = compute_ic(fac,10)
    iarr=np.array([x[1] for x in ic])
    print(f'=== {name} h10: n={len(ic)} IC={iarr.mean():.4f} ICIR={iarr.mean()/iarr.std():.4f} hit={(iarr>0).mean():.4f}')
    for h in [1,3,5,10,20]:
        i=compute_ic(fac,h); a=np.array([x[1] for x in i])
        print(f'   h{h}: IC={a.mean():.4f} ICIR={a.mean()/a.std():.4f}', end='')
    print()
    return iarr.mean(), iarr.mean()/iarr.std()

# Candidate B: 5d reversal (negative short-term momentum)
candB = -ret.rolling(5).sum().shift(1)
report('reversal_5d', candB)

# Candidate C: range position in 20d window  (1 - range/rolling... ) using close position
HIGH=high_df.rolling(20).max(); LOW=low_df.rolling(20).min()
range_pos = (close_df - LOW)/(HIGH - LOW + 1e-12)
range_pos = range_pos.shift(1)
report('range_pos_20d', range_pos)

# Candidate D: beta to equal-weight market (60d) using cov/var
mw = close_df.mean(axis=1)  # equal weight market proxy
mw_ret = mw.pct_change()
beta=[]
for s in watch:
    c = ret[s]
    cov = c.rolling(60).cov(mw_ret); var = mw_ret.rolling(60).var()
    beta.append(cov/var)
beta_df = pd.DataFrame(beta, index=watch).T.shift(1)
report('beta_ewmarket_60', beta_df)

# Candidate E: close vs 60d high drawdown distance (mean reversion to high)
dist_high = close_df/close_df.rolling(60).max().shift(1) - 1  # <=0, negative when below high
report('dist_high_60', dist_high)

# Candidate F: relative strength vs bond (asset momentum minus US10Y momentum)
bond = close_df['US10Y'].pct_change()
rs = ret - bond
# cross-section uses each asset's 10d return minus bond 10d return
rs10 = rs.rolling(10).sum().shift(1)
report('rs_vs_bond_10', rs10)