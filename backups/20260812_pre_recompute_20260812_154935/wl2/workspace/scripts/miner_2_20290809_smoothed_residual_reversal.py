import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,1200)
 if d is None or len(d)<100:d=get_index_daily_data(s,1200)
 if d is not None:px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); m=R.mean(axis=1)
bw,lh,smooth=40,5,3; rows=[]
# cache rolling beta, vol, and residual reversal for each asset/date
for s in P:
 x=R[s]; cov=x.rolling(bw).cov(m); vm=m.rolling(bw).var(); beta=cov/vm; vol=x.rolling(bw).std()
 resid=(x-beta*m).rolling(lh).sum(); P[s+'_sig']=-resid/(vol*np.sqrt(bw))
S=P[[s+'_sig' for s in U if s in P]].rename(columns=lambda x:x[:-4])
F=S.rolling(smooth,min_periods=1).mean()
for t in range(bw+lh+smooth,len(P)-1):
 f=F.iloc[t].dropna(); q=pd.concat([f,R.iloc[t+1].reindex(f.index)],axis=1).dropna()
 if len(q)>=8: rows.append((P.index[t],q.iloc[:,0].corr(q.iloc[:,1]),len(q)))
a=np.array([x[1] for x in rows]); ns=np.array([x[2] for x in rows])
print('PARAM',bw,lh,smooth,'dates',len(a),'avgN',round(ns.mean(),2),'coverage',round(ns.mean()/len(U),4),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)))
for lab,mask in [('pre2027',[x[0]<pd.Timestamp('2027-01-01') for x in rows]),('2027+',[x[0]>=pd.Timestamp('2027-01-01') for x in rows]),('2029+',[x[0]>=pd.Timestamp('2029-01-01') for x in rows])]:
 b=a[mask]; print(lab,'dates',len(b),'IC %.6f ICIR %.6f'%(b.mean(),b.mean()/b.std(ddof=1)))
print('UNIVERSE',len(U),'START',P.index.min(),'END',P.index.max())
