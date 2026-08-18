import os, glob
import numpy as np, pandas as pd
from scipy.stats import spearmanr

base='../persistent/stock_data'
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
    f=os.path.join(base,a+'.csv')
    if os.path.exists(f):
        d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date')
        px[a]=d['close'].astype(float)
prices=pd.DataFrame(px).sort_index()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].astype(float).sort_index()
# align and calculate signal using information through t; shift signal one day for execution
ret=prices.pct_change()
vol=ret.rolling(20,min_periods=15).std()
mom=prices/prices.shift(60)-1
# high-VIX regime: defensive low volatility; normal regime: risk-adjusted momentum
vix_med=vix.rolling(60,min_periods=40).median()
high=(vix>vix_med).reindex(prices.index).ffill()
signal=pd.DataFrame(index=prices.index,columns=assets,dtype=float)
signal.loc[~high.fillna(False),'SPX':] if False else None
for dt in prices.index:
    if pd.isna(high.get(dt, np.nan)): continue
    if bool(high.loc[dt]): signal.loc[dt]=(-vol.loc[dt])
    else: signal.loc[dt]=(mom.loc[dt]/vol.loc[dt])
signal=signal.shift(1)
# forward 10d returns, evaluated by date
fwd=prices.shift(-10)/prices-1
rows=[]
for dt in prices.index:
    x=signal.loc[dt]; y=fwd.loc[dt]
    ok=x.notna()&y.notna()
    if ok.sum()>=8:
        ic=spearmanr(x[ok],y[ok]).statistic
        rows.append((dt,ic,ok.sum(),high.loc[dt]))
r=pd.DataFrame(rows,columns=['date','ic','n','high_vix']).set_index('date')
for label,z in [('all',r),('normal',r[~r.high_vix]),('high_vix',r[r.high_vix]),('recent250',r.tail(250))]:
    ic=z.ic.mean(); sd=z.ic.std(ddof=1); icir=ic/sd*np.sqrt(252/10) if sd>0 else np.nan
    print(label,'dates',len(z),'avgN',z.n.mean() if len(z) else 0,'IC',round(ic,6),'ICIR',round(icir,6),'hit',round((z.ic>0).mean(),4) if len(z) else 0)
print('coverage',signal.notna().sum(axis=1).mean()/15,'last',prices.index.max(),'minN',r.n.min())
# rank turnover proxy
ranks=signal.rank(axis=1,pct=True); print('turnover',ranks.diff().abs().mean(axis=1).mean())
print('decay')
for h in [1,5,10,20]:
 f=prices.shift(-h)/prices-1; vals=[]
 for dt in prices.index:
  ok=signal.loc[dt].notna()&f.loc[dt].notna()
  if ok.sum()>=8: vals.append(spearmanr(signal.loc[dt,ok],f.loc[dt,ok]).statistic)
 print(h,len(vals),np.nanmean(vals))
