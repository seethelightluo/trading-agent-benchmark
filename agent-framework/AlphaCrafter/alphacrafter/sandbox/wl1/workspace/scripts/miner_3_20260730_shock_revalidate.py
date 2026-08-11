import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-07-15'); F={}; R={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]
 r=x.close.pct_change(); tr=(x.high-x.low)/x.close; base=tr.rolling(20,min_periods=15).median()
 F[s]=-(r*tr/(base+1e-12)); R[s]=r
f=pd.concat(F,axis=1); rr=pd.concat(R,axis=1)
# Daily IC and rank turnover, retaining date alignment
ics=[]; ranks=[]; ns=[]
for dt,row in f.iterrows():
 y=rr.shift(-1).loc[dt]
 z=pd.DataFrame({'f':row,'y':y}).dropna()
 if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:
  ics.append(spearmanr(z.f,z.y).statistic); ns.append(len(z)); ranks.append(z.f.rank(pct=True))
q=np.array(ics); turns=[]
for a,b in zip(ranks[:-1],ranks[1:]):
 ix=a.index.intersection(b.index)
 if len(ix)>=8: turns.append(np.abs(a[ix]-b[ix]).mean())
print('dates',len(q),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'turnover',round(np.mean(turns),4))
# independent artifact-like proxies for existing library factors
proxies={
 'clv':pd.concat({s:-(2*(pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').loc[:cut].close-pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').loc[:cut].low)/(pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').loc[:cut].high-pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').loc[:cut].low)-1) for s in U},axis=1),
 'rev5':-rr.rolling(5).sum(), 'mom20':rr.rolling(20).sum()/rr.rolling(20).std()
}
for n,p in proxies.items():
 a=f.stack().rename('f'); b=p.stack().rename('p'); z=pd.concat([a,b],axis=1).dropna(); print('corr',n,round(z.f.corr(z.p),4), 'pairs',len(z))
for y in sorted(set(pd.Index([d for d in f.index if d.year]).year)):
 z=q[[d.year==y for d in f.index if False]] if False else []
# regime from stored dates
print('regime', {y:round(np.mean([ics[i] for i,d in enumerate([d for d in f.index if d in []])]),5) for y in []})
print('regime not recomputed; full sample date-aligned')
