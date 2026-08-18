import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; px={}
for s in U:
 f=f'{base}/{s}.csv'
 if os.path.exists(f):
  d=pd.read_csv(f); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# Downside-adjusted medium-term trend: 40d return divided by downside deviation,
# lagged one day to avoid look-ahead; robustly rewards persistent gains with limited losses.
down=r.clip(upper=0).pow(2).rolling(40,min_periods=30).mean().pow(.5)
F=(P.pct_change(40).div(down.replace(0,np.nan))).shift(1)
vals=[]
for i in range(len(P)-20):
 z=pd.concat([F.iloc[i],P.iloc[i+10]/P.iloc[i]-1],axis=1).dropna()
 if len(z)>=8: vals.append((P.index[i],spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
a=np.array([x[1] for x in vals]); print('assets',len(px),'dates',len(P),'ic_dates',len(a),'avg_n',np.mean([x[2] for x in vals]))
for h in [1,3,5,10,20]:
 q=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q); print('horizon',h,'dates',len(q),'IC',np.nanmean(q),'ICIR',np.nanmean(q)/np.nanstd(q,ddof=1)*np.sqrt(len(q)),'hit',np.mean(q>0))
for n in [120,252,756,1260]:
 q=a[-n:]; print('recent',n,'IC',np.mean(q),'ICIR',np.mean(q)/np.std(q,ddof=1)*np.sqrt(len(q)),'hit',np.mean(q>0))
rank=F.rank(axis=1,pct=True); print('coverage',F.notna().mean().mean(),'turnover',np.nanmean(np.abs(rank-rank.shift(1)).mean(axis=1)))
