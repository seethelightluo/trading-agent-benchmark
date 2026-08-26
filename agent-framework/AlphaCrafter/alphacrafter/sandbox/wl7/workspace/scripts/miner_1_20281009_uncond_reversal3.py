import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];px={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None: px[s]=d.set_index(pd.to_datetime(d.date)).close
P=pd.DataFrame(px).sort_index();r=P.pct_change();ret=P/P.shift(3)-1;v=r.rolling(15).std();f=-(ret/(1+ret.abs()))/(v+1e-8)
I=[];D=[];C=[]
for i in range(len(P)-1):
 x=f.iloc[i];y=r.iloc[i+1];o=x.notna()&y.notna()
 if o.sum()>=8:
  z=x[o].corr(y[o],method='spearman')
  if np.isfinite(z):I.append(z);D.append(P.index[i]);C.append(o.mean())
I=np.array(I);print('dates',len(I),'avg_n',np.mean([((f.loc[d].notna()&r.loc[d+pd.Timedelta(days=1)].notna()).sum()) for d in D]),'coverage',np.mean(C),'IC',I.mean(),'ICIR',I.mean()/I.std(ddof=1),'hit',np.mean(I>0))
for a,b in [('2026','2027'),('2028','2028')]:
 z=I[(np.array(D)>=pd.Timestamp(a+'-01-01'))&(np.array(D)<=pd.Timestamp(b+'-12-31'))];print(a,b,len(z),z.mean(),z.mean()/z.std(ddof=1))
for h in [1,5,10]:
 q=[]
 for i in range(len(P)-h):
  x=f.iloc[i];y=P.iloc[i+h]/P.iloc[i]-1;o=x.notna()&y.notna()
  if o.sum()>=8:q.append(x[o].corr(y[o],method='spearman'))
 q=np.array(q);print('h',h,len(q),q.mean(),q.mean()/q.std(ddof=1))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20281009_uncond_reversal3_signal.csv',index=False)
