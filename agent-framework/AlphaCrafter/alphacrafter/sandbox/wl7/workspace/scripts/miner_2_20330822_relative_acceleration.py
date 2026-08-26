import pandas as pd, numpy as np
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2033-08-21'); P={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').loc[:cut]
 P[a]=d.close
P=pd.DataFrame(P); r=P.pct_change()
# Recent acceleration versus preceding 30d trend, lagged one completed session.
fac=(P.pct_change(10)-P.pct_change(30).shift(10)/3.0).shift(1)
print('cutoff',cut.date(),'dates',len(P),'assets',len(assets),'coverage %.4f'%fac.notna().mean().mean())
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; vals=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q): vals.append(q);ns.append(len(z))
 x=np.array(vals)
 print('H%d dates %d avgN %.2f minN %d IC %.6f ICIR %.6f hit %.4f thirds %s recent504 %.6f %.6f %.4f'%(
 h,len(x),np.mean(ns),min(ns),x.mean(),x.mean()/x.std(ddof=1),np.mean(x>0),[round(q.mean(),6) for q in np.array_split(x,3)],x[-504:].mean(),x[-504:].mean()/x[-504:].std(ddof=1),np.mean(x[-504:]>0)))
print('turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
out=[{'date':dt.date(),'asset':a,'signal':fac.loc[dt,a]} for dt in fac.index for a in assets]
pd.DataFrame(out).to_csv('scripts/miner_2_20330822_relative_acceleration_signal.csv',index=False)
