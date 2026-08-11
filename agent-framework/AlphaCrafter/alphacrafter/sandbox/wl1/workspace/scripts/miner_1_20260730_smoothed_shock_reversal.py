import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-07-15'); F={}; R={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]
 r=x.close.pct_change(); tr=(x.high-x.low)/x.close; base=tr.rolling(20,min_periods=15).median()
 shock=-(r*tr/(base+1e-12))
 F[s]=shock.rolling(3,min_periods=2).mean(); R[s]=r
f=pd.concat(F,axis=1); rr=pd.concat(R,axis=1)
for h in [1,3,5]:
  ics=[]; ns=[]; ranks=[]
  yret=rr.shift(-h).rolling(h).sum().shift(-(h-1))
  # equivalent next h sessions from t+1
  for dt,row in f.iterrows():
   z=pd.DataFrame({'f':row,'y':yret.loc[dt]}).dropna()
   if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:
    ics.append(spearmanr(z.f,z.y).statistic); ns.append(len(z)); ranks.append(z.f.rank(pct=True))
  q=np.array(ics); turns=[]
  for a,b in zip(ranks[:-1],ranks[1:]):
   ix=a.index.intersection(b.index)
   if len(ix)>=8: turns.append(np.abs(a[ix]-b[ix]).mean())
  print('h',h,'dates',len(q),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'turnover',round(np.mean(turns),4))
  for yr in range(2020,2027):
   v=[ics[i] for i,d in enumerate([dt for dt in f.index if dt in f.index][:len(ics)]) if False]
  # annual recompute directly
  out=[]
  for dt,row in f.iterrows():
   z=pd.DataFrame({'f':row,'y':yret.loc[dt]}).dropna()
   if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: out.append((dt,spearmanr(z.f,z.y).statistic))
  print('annual',[(yr,round(np.mean([v for d,v in out if d.year==yr]),4),sum(d.year==yr for d,v in out)) for yr in range(2020,2027) if any(d.year==yr for d,v in out)])
