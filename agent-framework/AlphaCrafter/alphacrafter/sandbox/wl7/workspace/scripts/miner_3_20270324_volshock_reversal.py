import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rets={}; fac={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None: continue
 d=d.copy();d.date=pd.to_datetime(d.date);d=d.set_index('date').sort_index(); c=d.close.astype(float)
 r=c.pct_change(); rv5=r.rolling(5,min_periods=5).std();rv60=r.rolling(60,min_periods=60).std()
 rets[s]=r;fac[s]=(-c.pct_change(5)*(rv5/rv60-1)).shift(1)
r=pd.DataFrame(rets); f=pd.DataFrame(fac);f=f.sub(f.median(axis=1),axis='index')
rows=[]
for h in [1,5,10,20]:
 vals=[]
 for dt in f.index:
  y=r.shift(-h).rolling(h).sum().loc[dt] # approximately forward sum; avoid future? r shift(-h) only one day wrong
  # use price forward return from each series via aligned r cumulative
  yy=[]
  for s in f.columns:
   # forward compounded h from dt+1..dt+h
   rr= r[s].loc[r[s].index>dt].head(h)
   yy.append(np.prod(1+rr.values)-1 if len(rr)==h else np.nan)
  z=pd.concat([f.loc[dt],pd.Series(yy,index=f.columns)],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=pd.Series(vals).dropna();rows.append((h,len(a),a.mean(),a.std(ddof=1),a.mean()/a.std(ddof=1)*np.sqrt(len(a)),(a>0).mean()))
valid=f.notna().sum(1)>=8
print('cutoff',f.index.max().date(),'assets',len(f.columns),'dates',len(f),'valid_dates',valid.sum(),'avg_n',f.notna().sum(1).loc[valid].mean(),'coverage',f.stack().notna().mean(),'turnover',f.rank(pct=True).diff().abs().mean(1).loc[valid].mean())
for x in rows: print('H',x[0],'n',x[1],'IC %.8f ICIR %.6f hit %.4f'%(x[2],x[4],x[5]))
for name,aa,bb in [('2020-22',2020,2022),('2023-24',2023,2024),('2025-27',2025,2027)]:
 vals=[]
 for dt in f.index:
  if not aa<=dt.year<=bb:continue
  yy=[]
  for s in f.columns:
   rr=r[s].loc[r[s].index>dt].head(1);yy.append(rr.iloc[0] if len(rr) else np.nan)
  z=pd.concat([f.loc[dt],pd.Series(yy,index=f.columns)],axis=1).dropna()
  if len(z)>=8:vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('REG',name,len(vals),np.nanmean(vals))
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_3_20270324_volshock_reversal_signal.csv',index=False)
