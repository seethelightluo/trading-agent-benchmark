import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p=pd.read_csv('../persistent/stock_data/'+s+'.csv'); p.date=pd.to_datetime(p.date); D[s]=p.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().ffill().loc[:'2033-10-16']; r=px.pct_change()
# Market-neutral 5-session reversal: remove each day's common cross-asset move,
# then reverse the asset's lagged 5d residual return and scale by lagged idiosyncratic volatility.
csmean=r.mean(axis=1); resid=r.sub(csmean,axis=0)
vol=resid.shift(1).rolling(40,min_periods=20).std()
f=(-resid.shift(1).rolling(5,min_periods=5).sum()).div(vol.replace(0,np.nan))
# strictly lag signal by using only completed observations through t-1
print('cutoff','2033-10-16','dates',len(px),'instruments',len(U),'coverage',round(f.notna().mean().mean(),4))
for h in [1,5,10,20]:
 a=[]; ns=[]; dates=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.x.nunique()>2:
   a.append(z.x.corr(z.y));ns.append(len(z));dates.append(px.index[i])
 q=pd.Series(a).dropna();
 print('H',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if h==10:
  n=len(q); print('thirds',*[round(q.iloc[j*n//3:(j+1)*n//3].mean(),6) for j in range(3)])
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
f.to_csv('scripts/miner_2_20331017_residual_reversal_signal.csv')
