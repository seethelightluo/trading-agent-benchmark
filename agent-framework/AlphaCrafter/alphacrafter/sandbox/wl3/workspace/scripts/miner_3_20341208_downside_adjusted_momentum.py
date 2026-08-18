import numpy as np
import pandas as pd
watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in watch:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); px[s]=d.drop_duplicates('date').set_index('date').close.astype(float)
px=pd.DataFrame(px).sort_index(); ret=px.pct_change(); neg=ret.clip(upper=0); dv=np.sqrt(neg.pow(2).rolling(20,min_periods=15).mean()); fac=px.pct_change(20)/dv.replace(0,np.nan); fac=fac.rank(axis=1,pct=True); fwd=px.shift(-10)/px-1
rows=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): rows.append((dt,c,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); scale=np.sqrt(252/10)
print('dates',len(r),'assets',len(px.columns),'avg_n',r.n.mean(),'coverage',r.n.sum()/(len(r)*15)); print('IC10',r.ic.mean(),'ICIR10',r.ic.mean()/r.ic.std(ddof=1)*scale,'hit',(r.ic>0).mean())
for n in [120,252]:
 q=r.tail(n); print('recent',n,'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1)*scale)
turn=fac.diff().abs().mean(axis=1).dropna(); print('turnover_rank_abs_change',turn.mean())
for lab,(a,b) in zip(['early','middle','late'],[(0,len(r)//3),(len(r)//3,2*len(r)//3),(2*len(r)//3,len(r))]):
 q=r.iloc[a:b]; print(lab,'dates',len(q),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1)*scale)
fac.index.name='date'; fac.reset_index().to_csv('scripts/miner_3_20341208_downside_adjusted_momentum_signal.csv',index=False); print('artifact ready')
