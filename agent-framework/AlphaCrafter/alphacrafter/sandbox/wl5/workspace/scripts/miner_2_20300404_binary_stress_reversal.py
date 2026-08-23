import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 f='../persistent/index_data/'+s+'.csv'
 if not os.path.exists(f): f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f): D[s]=pd.read_csv(f,parse_dates=['date']).drop_duplicates('date').set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); mom=p/p.shift(10)-1; vol=r.rolling(20).std()*np.sqrt(252)
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).drop_duplicates('date').set_index('date')['close'].astype(float).reindex(p.index).ffill()
med=vix.rolling(252,min_periods=126).median(); stress=(vix>med).astype(float)
# Binary macro interaction: modest reversal in calm, stronger reversal during stress.
sig=-(mom/vol).mul(0.75+0.75*stress,axis=0)
h=10; fwd=p.shift(-h)/p-1; vals=[];dates=[];ns=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if pd.notna(c): vals.append(c);dates.append(dt);ns.append(len(z))
q=pd.Series(vals,index=dates); print('horizon',h,'dates',len(q),'avg_instruments',round(float(np.mean(ns)),2),'IC %.8f ICIR %.8f hit %.4f'%(q.mean(),q.mean()/q.std(),(q>0).mean()))
for label,lo,hi in [('2020-24','2020','2025'),('2025-27','2025','2028'),('2028-30','2028','2030-04-05')]:
 t=q[(q.index>=lo)&(q.index<hi)];print(label,'dates',len(t),'IC %.8f ICIR %.8f hit %.4f'%(t.mean(),t.mean()/t.std(),(q.loc[t.index]>0).mean()))
print('coverage %.6f turnover %.6f instruments %d rows %d'%(sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean().mean(),len(D),int(sig.stack().size)))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20300404_binary_stress_reversal_signal.csv',index=False);print('artifact',len(out))
