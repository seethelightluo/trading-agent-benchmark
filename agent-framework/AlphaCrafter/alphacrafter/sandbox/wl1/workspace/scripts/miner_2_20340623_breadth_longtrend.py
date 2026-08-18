import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float) for s in U}).sort_index().ffill().loc[:'2034-06-23']; r=p.pct_change()
# Long-horizon trend aligned with cross-asset breadth: 60d return times signed 40d breadth, lagged.
t=p.pct_change(60); b=(p.pct_change(40)>0).mean(axis=1); sig=(t.mul(2*b-1,axis=0)).shift(1)
for h in [5,10,20,40]:
 f=p.shift(-h)/p-1;a=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]));ns.append(len(z))
 a=pd.Series(a);print(h,len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean())
# 10d regime
f=p.shift(-10)/p-1;a=[];ds=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
 if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]));ds.append(dt)
q=pd.Series(a,index=ds)
for x,y in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 z=q.loc[x:y];print(x,y,len(z),z.mean(),z.mean()/z.std(ddof=1))
print('turn',sig.rank(pct=True).diff().abs().mean().mean())
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20340623_breadth_longtrend_signal.csv',index=False)
