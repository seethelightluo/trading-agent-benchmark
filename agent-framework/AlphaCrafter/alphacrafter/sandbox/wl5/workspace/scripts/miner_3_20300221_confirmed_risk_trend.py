import numpy as np, pandas as pd, os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 f='../persistent/index_data/'+s+'.csv'
 if not os.path.exists(f): f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f): D[s]=pd.read_csv(f,parse_dates=['date']).sort_values('date').drop_duplicates('date').set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); ret=p.pct_change(); mom60=p/p.shift(60)-1; vol60=ret.rolling(60).std()*np.sqrt(252); mom20=p/p.shift(20)-1; sig=(mom60/vol60)*(1+0.5*np.sign(mom60)*np.sign(mom20))
for h in [5,10,20]:
 vals=[]; dates=[]; cs=[]; fwd=p.shift(-h)/p-1
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); cs.append(len(z))
 q=pd.Series(vals,index=dates).dropna(); print(h,'dates',len(q),'IC %.8f ICIR %.8f hit %.4f nmean %.2f'%(q.mean(),q.mean()/q.std(),(q>0).mean(),np.mean(cs)))
 for label,lo,hi in [('2020-24','2020','2025'),('2025-27','2025','2028'),('2028-30','2028','2030-02-22')]:
  t=q[(q.index>=lo)&(q.index<hi)]; print(' ',label,len(t),'%.8f %.8f'%(t.mean(),t.mean()/t.std()))
print('coverage',sig.notna().mean().mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20300221_confirmed_risk_trend_signal.csv',index=False); print('artifact rows',len(out),'instruments',len(D))
