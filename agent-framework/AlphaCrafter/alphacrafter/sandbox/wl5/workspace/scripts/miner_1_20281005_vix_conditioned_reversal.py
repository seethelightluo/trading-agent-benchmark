import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2028-10-05')
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date'); px[s]=d['close']
P=pd.DataFrame(px).sort_index().loc[:cutoff]; macro=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date'); v=macro['close'].reindex(P.index).ffill(); vs=v.rolling(20).mean(); stress=(v/vs-1).clip(-.5,1.5)
r=P.pct_change(5); f=-r.div(P.pct_change().rolling(20).std()*np.sqrt(252)).mul(1+stress.clip(lower=0))
rows=[]
for i in range(25,len(P)-10):
 z=pd.concat([f.iloc[i],P.pct_change(10).iloc[i+10]],axis=1).dropna()
 if len(z)>=8: rows.append((P.index[i],z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
R=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('dates',len(R),'avg_n',R.n.mean(),'coverage',R.n.sum()/(len(R)*15)); print('IC',R.ic.mean(),'ICIR',R.ic.mean()/R.ic.std(ddof=1),'hit',(R.ic>0).mean())
for a,b in [('2020','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31'),('2028-01-01','2028-10-04')]:
 q=R.loc[a:b].ic; print('regime',a,b,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
for h in [5,10,20]:
 rr=[]
 for i in range(25,len(P)-h):
  z=pd.concat([f.iloc[i],P.pct_change(h).iloc[i+h]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(rr),np.nanmean(rr)/np.nanstd(rr,ddof=1))
R.to_csv('scripts/miner_1_20281005_vix_conditioned_reversal_signal.csv')
