import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-07-15')
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@END').drop_duplicates('date').set_index('date'); return d.close
p=pd.DataFrame({s:load(s) for s in U}).sort_index(); r=p.pct_change(); m=r['SPX']
# downside beta: covariance with SPX only on its negative-return sessions, normalized by SPX downside variance; defensive = negative beta
fac=pd.DataFrame(index=r.index,columns=U,dtype=float)
for i,dt in enumerate(r.index):
 if i<60: continue
 sl=slice(i-60,i); x=m.iloc[sl]; mask=x<0
 if mask.sum()<12: continue
 den=(x[mask]**2).sum()
 if den<=0: continue
 for s in U:
  y=r[s].iloc[sl]; ok=mask & y.notna()
  if ok.sum()>=12: fac.loc[dt,s]=-(y[ok]*x[ok]).sum()/((x[ok]**2).sum())
# cross-sectional IC and metrics
for h in [1,5,10]:
 fw=p.pct_change(h).shift(-h); vals=[]; ns=[]
 for dt in fac.index:
  z=pd.DataFrame({'f':fac.loc[dt],'y':fw.loc[dt]}).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: vals.append(spearmanr(z.f,z.y).statistic); ns.append(len(z))
 a=np.array(vals); print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  # reconstruct date list in same order
  ds=[dt for dt in fac.index if len(pd.DataFrame({'f':fac.loc[dt],'y':fw.loc[dt]}).dropna())>=8 and pd.DataFrame({'f':fac.loc[dt],'y':fw.loc[dt]}).dropna().f.nunique()>1 and pd.DataFrame({'f':fac.loc[dt],'y':fw.loc[dt]}).dropna().y.nunique()>1]
  q=np.array([v for d,v in zip(ds,a) if lo<=d.year<=hi]); print(' regime',lo,hi,'n',len(q),'IC',round(q.mean(),6) if len(q) else None)
# rank turnover, coverage
rank=fac.rank(axis=1,pct=True); print('coverage',round(fac.notna().mean().mean(),4),'turnover',round(rank.diff().abs().mean(axis=1).mean(),4),'period',p.index.min().date(),p.index.max().date())
# pooled correlations with production-like CLV, reversal, momentum
rows=[]
for s in U:
 clv=2*(p[s]-pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').low.reindex(p.index))/(pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').high.reindex(p.index)-pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').low.reindex(p.index))-1
 for dt in fac.index:
  z=[fac.loc[dt,s],-clv.loc[dt],-r[s].rolling(5).sum().loc[dt],(r[s].rolling(20).sum()/r[s].rolling(20).std()).loc[dt]]
  if all(np.isfinite(z)): rows.append(z)
c=np.array(rows); print('pooled_corr',*[round(spearmanr(c[:,0],c[:,j]).statistic,4) for j in [1,2,3]])
