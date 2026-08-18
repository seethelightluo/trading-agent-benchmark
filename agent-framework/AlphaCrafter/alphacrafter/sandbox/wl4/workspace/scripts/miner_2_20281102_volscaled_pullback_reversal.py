import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv')); d.date=pd.to_datetime(d.date); px[s]=d.sort_values('date').set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().ffill(); R=P.pct_change(); vol=R.rolling(20).std()*np.sqrt(20)
F=(-(P.pct_change(5))/vol).clip(-8,8)
rows=[]; ranks=[]
for i in range(20,len(P)-10):
 dt=P.index[i]
 if dt>pd.Timestamp('2028-11-01'): break
 z=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+10]/P.iloc[i]-1).rename('r')],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.f,z.r).statistic,len(z))); ranks.append(F.iloc[i].rank(pct=True))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for label,y in [('all',x),('recent250',x.tail(250)),('online',x[x.index>='2026-07-16'])]:
 mean=y.ic.mean(); sd=y.ic.std(ddof=1); print(label,'dates',len(y),'avg_n',round(y.n.mean(),2),'IC',round(mean,5),'ICIR_ann',round(mean/sd*np.sqrt(252),5),'ICIR_raw',round(mean/sd,5),'hit',round((y.ic>0).mean(),3))
print('decay',end=' ')
for h in [1,5,10,20]:
 a=[]
 for i in range(20,len(P)-h):
  if P.index[i]>pd.Timestamp('2028-11-01'): break
  z=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('r')],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.f,z.r).statistic)
 print(h,round(np.nanmean(a),5),end='; ')
turn=np.mean([np.mean(np.abs(ranks[i]-ranks[i-1])) for i in range(1,len(ranks))])
print('\ncoverage',round(F.notna().mean().mean(),5),'turnover_proxy',round(turn,5),'dates',len(x),'last',x.index[-1].date())
