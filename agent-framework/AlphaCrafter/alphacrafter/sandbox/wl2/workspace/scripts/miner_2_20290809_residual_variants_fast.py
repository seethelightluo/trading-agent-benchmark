import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,1200)
 if d is None or len(d)<100:d=get_index_daily_data(s,1200)
 if d is not None:px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); m=R.mean(axis=1)
for bw,lh,sm in [(40,3,3),(40,5,5),(60,5,3),(60,5,5)]:
 S={}
 for s in P:
  beta=R[s].rolling(bw).cov(m)/m.rolling(bw).var(); vol=R[s].rolling(bw).std()
  S[s]=-(R[s]-beta*m).rolling(lh).sum()/(vol*np.sqrt(bw))
 F=pd.DataFrame(S).rolling(sm,min_periods=1).mean(); vals=[]; ns=[]
 for t in range(bw+lh+sm,len(P)-1):
  q=pd.concat([F.iloc[t],R.iloc[t+1]],axis=1).dropna()
  if len(q)>=8: vals.append(q.iloc[:,0].corr(q.iloc[:,1])); ns.append(len(q))
 a=np.array(vals);print('PARAM',bw,lh,sm,'dates',len(a),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)))
