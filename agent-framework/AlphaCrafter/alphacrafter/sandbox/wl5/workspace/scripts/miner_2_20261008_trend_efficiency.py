import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-10-07')
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date'); return d
D={s:load(s) for s in U}; p=pd.DataFrame({s:D[s].close for s in U}).sort_index(); r=p.pct_change()
# trend-efficiency: net 20d return divided by total absolute daily movement, lagged one day by construction at decision
fac=pd.DataFrame(index=p.index,columns=U,dtype=float)
for s in U:
 rr=r[s]; fac[s]=(rr.rolling(20,min_periods=18).sum()/rr.abs().rolling(20,min_periods=18).sum()).shift(1)
# IC with forward close-to-close returns
for h in [1,5,10,20]:
 fw=p.pct_change(h).shift(-h); vals=[]; ns=[]; ds=[]
 for dt in fac.index:
  z=pd.DataFrame({'f':fac.loc[dt],'y':fw.loc[dt]}).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:
   vals.append(spearmanr(z.f,z.y).statistic); ns.append(len(z)); ds.append(dt)
 a=np.array(vals); print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  q=a[[lo<=d.year<=hi for d in ds]]; print(' regime',lo,hi,'n',len(q),'IC',round(q.mean(),6) if len(q) else None,'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)
rank=fac.rank(axis=1,pct=True); print('coverage',round(fac.notna().mean().mean(),4),'turnover',round(rank.diff().abs().mean(axis=1).mean(),4),'period',p.index.min().date(),p.index.max().date())
# correlation to common factors (pooled, comparable simple constructions)
rows=[]
for s in U:
 rr=r[s]; clv=2*(D[s].close-D[s].low)/(D[s].high-D[s].low)-1
 for dt in fac.index:
  z=[fac.loc[dt,s],-clv.reindex(p.index).loc[dt],-rr.rolling(5).sum().loc[dt],(rr.rolling(20).sum()/rr.rolling(20).std()).loc[dt]]
  if all(np.isfinite(z)): rows.append(z)
c=np.array(rows); print('pooled_corr',*[round(spearmanr(c[:,0],c[:,j]).statistic,4) for j in [1,2,3]])
