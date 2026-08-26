import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float)
 D[s]=x
px=pd.DataFrame(D).sort_index().ffill().loc[:'2034-01-19']; r=px.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.astype(float).reindex(px.index).ffill()
# Candidate: continuous stress-weighted 5d reversal. All inputs are shifted one day.
lag5=r.rolling(5,min_periods=5).sum().shift(1)
vol=r.rolling(40,min_periods=30).std().shift(1)
vs=v.pct_change(5); vp=vs.rolling(120,min_periods=60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).shift(1)
bread=(-r.rolling(20,min_periods=15).sum()).clip(lower=0).mean(axis=1).shift(1)
stress=(vp.clip(lower=0)*bread).clip(lower=0)
f=(-lag5/(vol+1e-8)).mul(stress,axis=0)
f=f.sub(f.median(axis=1),axis=0)
for h in [1,5,10,20]:
 a=[];ns=[]
 for i in range(len(px)-h):
  y=px.iloc[i+h]/px.iloc[i+1]-1
  z=pd.concat([f.iloc[i].rename('x'),y.rename('y')],axis=1).dropna()
  if len(z)>=8 and z.x.nunique()>1 and z.y.nunique()>1:
   a.append(z.x.corr(z.y,method='spearman'));ns.append(len(z))
 q=pd.Series(a).dropna(); print('H',h,'IC %.6f ICIR %.6f hit %.3f dates %d avgN %.2f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),len(q),np.mean(ns)))
print('range',px.index.min().date(),px.index.max().date(),'dates',len(px),'assets',len(U),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean(),'active',float((stress>0).mean()))
f.astype(float).to_csv('scripts/miner_1_20340123_continuous_stress_reversal_signal.csv',index_label='date')
# thirds for 10d stability
q=[]
for i in range(len(px)-10):
 z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+10]/px.iloc[i+1]-1).rename('y')],axis=1).dropna()
 if len(z)>=8 and z.x.nunique()>1:q.append((px.index[i],z.x.corr(z.y,method='spearman')))
qq=pd.Series(dict(q)); print('thirds',*[round(x,6) for x in np.array_split(qq,3) if len(x) for x in [x.mean()]])
