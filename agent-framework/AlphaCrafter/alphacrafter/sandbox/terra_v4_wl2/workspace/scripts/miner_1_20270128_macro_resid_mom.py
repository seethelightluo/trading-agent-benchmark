import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
root=Path('../persistent/stock_data')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
macro=['DXY','VIX']
allx={}
for s in syms:
 d=pd.read_csv(root/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].pct_change(); allx[s]=d
for s in macro:
 d=pd.read_csv(Path('../persistent/index_data')/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].pct_change(); allx[s]=d
r=pd.DataFrame(allx).sort_index(); r=r.loc[:'2027-01-27']
# residual 20d return after rolling 60d OLS on daily returns against SPX,DXY,VIX,US10Y
# signal at t uses through t; forward t+1 close return
out=[]
for s in syms:
 vals=[]
 for i in range(len(r)):
  if i<60: vals.append(np.nan); continue
  y=r[s].iloc[i-59:i+1]; X=r[['SPX','DXY','VIX','US10Y']].iloc[i-59:i+1]
  z=pd.concat([y,X],axis=1).dropna()
  if len(z)<45: vals.append(np.nan); continue
  A=np.column_stack([np.ones(len(z)),z.iloc[:,1:].values]); b=np.linalg.lstsq(A,z.iloc[:,0].values,rcond=None)[0]
  resid=z.iloc[:,0].values-A@b
  vals.append(np.sum(resid[-20:]))
 out.append(pd.Series(vals,index=r.index,name=s))
f=pd.concat(out,axis=1); fw=r[syms].shift(-1)
ics=[]; used=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); used.append(dt)
ics=pd.Series(ics,index=used).dropna(); print('range',ics.index.min(),ics.index.max()); print('dates',len(ics),'avg_names',np.mean([pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna().shape[0] for d in used]))
print('IC %.8f ICIR %.8f hit %.4f cov %.4f turnover %.4f'%(ics.mean(),ics.mean()/ics.std(ddof=1),np.mean(ics>0),f.notna().sum().sum()/(len(f)*15),f.rank(axis=1).diff().abs().mean().mean()/14))
for yr in [2020,2021,2022,2023,2024,2025,2026,2027]:
 q=ics[ics.index.year==yr]; print('year',yr,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
for h in [5,10]:
 fw2=r[syms].shift(-h).div(r[syms])-1; ii=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw2.loc[dt]],axis=1).dropna()
  if len(z)>=8: ii.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 ii=pd.Series(ii).dropna(); print('h',h,'IC',ii.mean(),'ICIR',ii.mean()/ii.std(ddof=1))
# save signal artifact
sig=f.copy(); sig.index.name='date'; sig.reset_index().to_csv('../persistent/factor_signals_miner_1_20270128_macro_resid_mom.csv',index=False)
