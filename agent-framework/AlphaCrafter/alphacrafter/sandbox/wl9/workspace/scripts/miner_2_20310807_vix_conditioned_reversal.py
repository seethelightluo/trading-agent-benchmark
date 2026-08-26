import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; cutoff=pd.Timestamp('2031-08-07')
px={}
for s in U:
 p=os.path.join(base,s+'.csv')
 if os.path.exists(p):
  d=pd.read_csv(p); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close.rename(s)
prices=pd.concat(px.values(),axis=1).sort_index(); prices=prices.loc[:cutoff]
macro=pd.read_csv('../persistent/index_data/VIX.csv'); macro.date=pd.to_datetime(macro.date); v=macro.set_index('date').close.reindex(prices.index).ffill()
r=np.log(prices).diff(); cs=r.sub(r.mean(axis=1),axis=0)
# Contrarian residual momentum, amplified smoothly by high VIX percentile (strictly lagged)
rev=-(cs.rolling(60,min_periods=40).sum())
vixscore=v.rolling(252,min_periods=100).rank(pct=True)
signal=rev.mul(0.75+0.75*vixscore,axis=0).shift(1)
rows=[]
for h in [5,10,20,40]:
 f=prices.pct_change(h).shift(-h)
 vals=[]
 for dt in signal.index:
  a=signal.loc[dt]; b=f.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
  if len(z)>=8: vals.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 x=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
 print(h,'dates',len(x),'avg_n',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,4),'IC',round(x.ic.mean(),6),'ICIR',round(x.ic.mean()/x.ic.std(ddof=1),6),'hit',round((x.ic>0).mean(),4))
 for name,lo,hi in [('2024-26','2024','2026-12-31'),('2027-29','2027','2029-12-31'),('2030','2030','2030-12-31'),('2031','2031','2031-08-07')]:
  q=x.loc[pd.Timestamp(lo):pd.Timestamp(hi),'ic'];
  if len(q): print(' ',name,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
# artifact at admission horizon
f=prices.pct_change(20).shift(-20); out=[]
for dt in signal.index:
 z=pd.concat([signal.loc[dt],f.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  for s in z.index: out.append([dt,s,signal.loc[dt,s]])
pd.DataFrame(out,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20310807_vix_conditioned_reversal_signal.csv',index=False)
