import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; cutoff=pd.Timestamp('2031-08-21')
px={}
for s in U:
 p=os.path.join(base,s+'.csv')
 if os.path.exists(p):
  d=pd.read_csv(p); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close.rename(s)
prices=pd.concat(px.values(),axis=1).sort_index().loc[:cutoff]
r=np.log(prices).diff()
down=r.clip(upper=0).rolling(60,min_periods=40).sum()
negfrac=(r<0).rolling(60,min_periods=40).mean()
# RMS downside magnitude avoids sparse-negative rolling std failure
downvol=np.sqrt((r.clip(upper=0)**2).rolling(60,min_periods=40).mean())
rank1=down.rank(axis=1,pct=True); rank2=negfrac.rank(axis=1,pct=True); rank3=downvol.rank(axis=1,pct=True)
signal=-(0.5*rank1+0.3*rank2+0.2*rank3).shift(1)
for h in [5,10,20,40]:
 f=prices.pct_change(h).shift(-h); vals=[]
 for dt in signal.index:
  z=pd.concat([signal.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 x=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
 print(h,'dates',len(x),'avg_n',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,4),'IC',round(x.ic.mean(),6),'ICIR',round(x.ic.mean()/x.ic.std(ddof=1),6),'hit',round((x.ic>0).mean(),4))
 for name,lo,hi in [('2024-26','2024','2026-12-31'),('2027-29','2027','2029-12-31'),('2030','2030','2030-12-31'),('2031','2031','2031-08-21')]:
  q=x.loc[pd.Timestamp(lo):pd.Timestamp(hi),'ic']
  if len(q): print(' ',name,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
 if h==20:
  out=[]
  for dt in signal.index:
   z=pd.concat([signal.loc[dt],f.loc[dt]],axis=1).dropna()
   if len(z)>=8:
    for s in z.index: out.append([dt,s,signal.loc[dt,s]])
  pd.DataFrame(out,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20310821_downside_persistence_quality_signal.csv',index=False)
