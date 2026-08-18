import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
 f=f'{base}/{s}.csv'
 if os.path.exists(f):
  d=pd.read_csv(f); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
vix=pd.read_csv('../persistent/index_data/VIX.csv'); vix.date=pd.to_datetime(vix.date); v=vix.set_index('date').close.reindex(P.index).ffill()
# candidate: 20d relative momentum, dampened in high volatility, lagged
mom=P.pct_change(20); vm=v.rolling(60).median(); stress=(v/vm).clip(0.5,2.0)
# reversal of volatility regime: reward momentum in calm and reversal in stress (interpretable)
F=(mom.div(stress,axis=0)).shift(1)
rows=[]
for h in [1,3,5,10,20]:
  vals=[]
  for i in range(len(P)-h):
   x=F.iloc[i]; y=P.iloc[i+h]/P.iloc[i]-1; z=pd.concat([x,y],axis=1).dropna()
   if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  a=np.array(vals); ic=np.nanmean(a); sd=np.nanstd(a,ddof=1); ir=ic/sd*np.sqrt(len(a)) if sd else np.nan
  rows.append((h,len(a),ic,ir,np.mean(a>0)))
print('assets',len(px),'dates',len(P),'avg cross-section',np.mean([F.iloc[i].notna().sum() for i in range(len(P))]))
for x in rows: print('horizon dates IC ICIR hit',x)
# recent 252 10d
vals=[]
h=10
for i in range(len(P)-h):
 z=pd.concat([F.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
 if len(z)>=8: vals.append((P.index[i],spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
for n in [120,252,756,1260]:
 a=np.array([q[1] for q in vals[-n:]]); print('recent',n,'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1)*np.sqrt(len(a)),'IC',np.nanmean(a))
# turnover rank changes
rank=F.rank(axis=1,pct=True); print('turnover',np.nanmean(np.abs(rank-rank.shift(1)).mean(axis=1)))
