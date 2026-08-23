import pandas as pd, numpy as np, os, warnings
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; cutoff=pd.Timestamp('2027-02-10')
prices={}; rets={}
for a in assets:
 p=f'{base}/{a}.csv'
 if not os.path.exists(p): continue
 d=pd.read_csv(p); d.date=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date').loc[:cutoff]
 prices[a]=d.close; rets[a]=d.close.pct_change()
R=pd.DataFrame(rets); mom=R.rolling(20,min_periods=15).sum(); med=mom.median(axis=1); vol=R.rolling(20,min_periods=15).std(); fac=(mom.sub(med,axis=0)/vol).shift(1)
def calc(h):
 rows=[]
 for a in assets:
  if a not in prices: continue
  px=prices[a]; f=fac[a].reindex(px.index); fr=px.shift(-h)/px-1
  rows.append(pd.DataFrame({'date':px.index,'f':f.values,'r':fr.values}))
 dd=pd.concat(rows).dropna(); obs=[]
 for dt,g in dd.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1:
   ic=spearmanr(g.f,g.r).statistic
   if np.isfinite(ic): obs.append((dt,ic,len(g)))
 return pd.DataFrame(obs,columns=['date','ic','n'])
for h in [1,3,5,10]:
 z=calc(h); print(h,'dates',len(z),'avg_n',round(z.n.mean(),2),'coverage',round(z.n.sum()/(len(z)*15),4),'IC',round(z.ic.mean(),5),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),5),'hit',round((z.ic>0).mean(),4))
z=calc(1); print('period',z.date.min().date(),z.date.max().date())
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2027-02-10')]:
 q=z[(z.date>=lo)&(z.date<=hi)]
 if len(q): print('regime',lo,hi,'dates',len(q),'IC',round(q.ic.mean(),5),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),5),'hit',round((q.ic>0).mean(),4))
ranks=fac.rank(axis=1,pct=True); print('turnover',round(ranks.diff().abs().mean(axis=1).mean(),5))
