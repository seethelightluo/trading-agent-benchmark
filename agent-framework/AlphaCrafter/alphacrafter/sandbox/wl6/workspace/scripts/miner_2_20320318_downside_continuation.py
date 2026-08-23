import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-03-17'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 px[s]=d.close.astype(float)
p=pd.concat(px,axis=1).sort_index().loc[:cut]; r=p.pct_change()
# Trend continuation adjusted by downside risk: positive 20d trend is preferred only when
# the prior 60d downside deviation is modest. All values are lagged at each decision date.
down=r.clip(upper=0).rolling(60,min_periods=40).std()*np.sqrt(60)
sig=(p/p.shift(20)-1)/(down+1e-12)
print('candidate=downside_continuation_20d dates=%d instruments=%d first=%s last=%s'%(len(p),len(U),p.index.min().date(),p.index.max().date()),flush=True)
for h in [5,10,20]:
 f=p.shift(-h)/p-1; z=[]; ns=[]; ds=[]
 for dt in sig.index:
  a=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   q=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if pd.notna(q): z.append(q);ns.append(len(a));ds.append(dt)
 z=pd.Series(z,index=ds); ir=z.mean()/z.std(ddof=1)*np.sqrt(len(z))
 print('h=%d dates=%d avg_n=%.2f coverage=%.5f IC=%+.8f ICIR=%+.6f hit=%.4f'%(h,len(z),np.mean(ns),np.mean(ns)/15,z.mean(),ir,(z>0).mean()),flush=True)
 print('regimes',z.groupby(z.index.year).agg(['mean','count']).round(6).to_dict('index'),flush=True)
print('turnover=%.6f'%sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),flush=True)
