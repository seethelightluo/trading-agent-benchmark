import pandas as pd, numpy as np, glob, os
from scipy.stats import pearsonr
files=glob.glob('../persistent/stock_data/*.csv')
use=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in use:
 f='../persistent/stock_data/'+s+'.csv'
 d=pd.read_csv(f); d['date']=pd.to_datetime(d['date']); px[s]=d.set_index('date')['close']
prices=pd.DataFrame(px).sort_index(); prices=prices.loc[:'2027-02-25']; ret=prices.pct_change(); mkt=ret.mean(axis=1)
# beta of each asset to contemporaneous equal-weight benchmark, signal is negative beta
beta=ret.rolling(60,min_periods=40).cov(mkt).div(mkt.rolling(60,min_periods=40).var(),axis=0)
sig=-beta
fwd=prices.shift(-1).div(prices)-1
rows=[]
for dt in sig.index:
 x=sig.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ic=z.iloc[:,0].corr(z.iloc[:,1]); rows.append((dt,ic,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); r=r.replace([np.inf,-np.inf],np.nan).dropna()
print('dates',len(r),'avg_n',r.n.mean(),'period',r.index.min(),r.index.max())
for label,sub in [('all',r),('2020_22',r.loc['2020':'2022']),('2023_24',r.loc['2023':'2024']),('2025_26',r.loc['2025':'2026']),('online',r.loc['2026-07-16':])]:
 if len(sub): print(label,'n',len(sub),'IC',sub.ic.mean(),'ICIR',sub.ic.mean()/sub.ic.std(ddof=1),'hit',(sub.ic>0).mean())
# decay horizons
for h in [1,3,5,10]:
 yy=prices.shift(-h).div(prices)-1; rr=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 rr=pd.Series(rr).dropna(); print('h',h,'IC',rr.mean(),'ICIR',rr.mean()/rr.std(ddof=1),'n',len(rr))
# turnover rank proxy
rank=sig.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna(); print('turn',turn.mean(),'coverage', (sig.notna().sum(axis=1)/15).mean())
out=sig.stack().rename('signal').reset_index(); out.columns=['date','asset','signal']; out.to_csv('../persistent/factor_signals_miner_1_20270225_lowbeta60_new.csv',index=False)
