import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s,days=5000)
    if x is not None and len(x)>150:
        x=x[['date','close']].copy(); x.date=pd.to_datetime(x.date)
        D[s]=x.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff()
# Candidate: medium-term trend continuation, penalized by downside risk and conditioned on broad participation.
m20=np.log(p/p.shift(20)); m60=np.log(p/p.shift(60)); dn=r.where(r<0,0).rolling(40).std(); rv=r.rolling(20).std()
bread=(m20>0).mean(axis=1).rolling(10).mean()
# Smooth participation multiplier, avoids a binary regime lookup.
gate=(0.65+0.70*bread).clip(0.65,1.35)
f=(0.55*m20+0.45*m60)/(dn+0.5*rv+1e-8)*gate.values[:,None]
# one-day lag in signal evaluation
f=f.shift(1)
rows=[]
for h in [1,5,10,20]:
 vals=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i],np.log(p.iloc[i+h]/p.iloc[i])],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(vals).dropna(); rows.append({'h':h,'obs':len(q),'ic':q.mean(),'icir':q.mean()/q.std(ddof=1),'hit':(q>0).mean()})
print('dates',len(p),'assets',len(D),'avg_assets',f.notna().sum(axis=1).mean(),'rows',rows)
rank=f.rank(axis=1,pct=True); print('coverage',f.notna().sum(axis=1).mean()/len(U),'turnover',rank.diff().abs().mean(axis=1).mean())
# regime thirds at admission horizon
h=10
for a,b in [(0,len(p)//3),(len(p)//3,2*len(p)//3),(2*len(p)//3,len(p))]:
 vals=[]
 for i in range(a,min(b,len(p)-h)):
  z=pd.concat([f.iloc[i],np.log(p.iloc[i+h]/p.iloc[i])],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(vals).dropna(); print('regime',p.index[a].date(),p.index[min(b-1,len(p)-1)].date(),len(q),q.mean(),q.mean()/q.std(ddof=1))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20311030_participation_trend_signal.csv',index=False)
