import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,1200)
 if d is None or len(d)<100:d=get_index_daily_data(s,1200)
 if d is not None:px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index();R=P.pct_change();m=R.mean(axis=1)
# use precomputed rolling beta/residual sums; signal is trailing mean of daily residual reversal scores
for bw,lh,smooth in [(40,5,3),(60,5,3)]:
 rows=[]
 for t in range(bw+lh+smooth,len(P)-1):
  sig=[]
  for j in range(t-smooth+1,t+1):
   vals={}
   for s in P:
    x=R[s].iloc[j-bw+1:j+1]; y=m.iloc[j-bw+1:j+1]; z=pd.concat([x,y],axis=1).dropna()
    if len(z)<20 or z.iloc[:,1].var()<=1e-12:continue
    beta=z.iloc[:,0].cov(z.iloc[:,1])/z.iloc[:,1].var(); vol=z.iloc[:,0].std()
    resid=(R[s].iloc[j-lh+1:j+1]-beta*m.iloc[j-lh+1:j+1]).sum()
    if vol>1e-8:vals[s]=-resid/(vol*np.sqrt(bw))
   sig.append(pd.Series(vals))
  f=pd.concat(sig,axis=1).mean(axis=1).dropna();q=pd.concat([f,R.iloc[t+1].reindex(f.index)],axis=1).dropna()
  if len(q)>=8:rows.append((P.index[t],q.iloc[:,0].corr(q.iloc[:,1]),len(q)))
 a=np.array([x[1] for x in rows]);ns=np.array([x[2] for x in rows])
 print('bw/lh/smooth',bw,lh,smooth,'dates',len(a),'avgN',round(ns.mean(),2),'coverage',round(ns.mean()/len(P.columns),4),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)))
 for lab,cut in [('2028+',pd.Timestamp('2028-01-01')),('2029+',pd.Timestamp('2029-01-01'))]:
  b=a[[x[0]>=cut for x in rows]]; print(lab,'dates',len(b),'IC %.6f ICIR %.6f'%(b.mean(),b.mean()/b.std(ddof=1)))
