import numpy as np, pandas as pd, os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 f='../persistent/index_data/'+s+'.csv'
 if not os.path.exists(f): f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f): D[s]=pd.read_csv(f,parse_dates=['date']).sort_values('date').drop_duplicates('date').set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); m60=p/p.shift(60)-1; v60=r.rolling(60).std()*np.sqrt(252); m20=p/p.shift(20)-1
base=(m60/v60)*(1+0.5*np.sign(m60)*np.sign(m20)); sig=-base
h=20; fwd=p.shift(-h)/p-1; vals=[]; dates=[]; ns=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); ns.append(len(z))
q=pd.Series(vals,index=dates).dropna(); ic=q.mean(); icir=ic/q.std(); print('horizon',h,'dates',len(q),'avg_instruments',np.mean(ns),'IC %.8f ICIR %.8f hit %.4f'%(ic,icir,(q>0).mean()))
for label,lo,hi in [('2020-24','2020','2025'),('2025-27','2025','2028'),('2028-30','2028','2030-03-08')]:
 t=q[(q.index>=lo)&(q.index<hi)]; print(label,'dates',len(t),'IC %.8f ICIR %.8f'%(t.mean(),t.mean()/t.std()))
print('coverage %.6f turnover %.6f instruments %d rows %d'%(sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean().mean(),len(D),int(sig.stack().size)))
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20300307_inverse_risk_trend_signal.csv',index=False)
print('artifact',len(out))
