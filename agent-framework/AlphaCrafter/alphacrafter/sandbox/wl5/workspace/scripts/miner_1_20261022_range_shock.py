import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-10-21')
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']); return d[d.date<=END].drop_duplicates('date').sort_values('date').set_index('date')
D={s:load(s) for s in U}; sig={}; ret={}
for s,d in D.items():
 r=d.close.pct_change(); ret[s]=r
 tr=(d.high-d.low)/d.close.shift(1)
 # shock reversal: negative return after unusually large range, normalized by rolling typical range
 sig[s]=-(r*tr/tr.rolling(20,min_periods=10).median())
def report(h):
 rows=[]
 for s in U:
  y= D[s].close.pct_change(h).shift(-h)
  z=pd.concat([sig[s],y],axis=1).dropna()
  rows += [(dt,s,float(a),float(b)) for dt,a,b in z.itertuples()]
 q=pd.DataFrame(rows,columns=['date','s','f','y']); ics=[]; ns=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1: ics.append((dt,spearmanr(g.f,g.y).statistic));ns.append(len(g))
 a=np.array([v for _,v in ics]); print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  x=np.array([v for d,v in ics if lo<=d.year<=hi]);print('regime',lo,hi,'n',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
for h in [1,5,10]: report(h)
# coverage and rank turnover
wide=pd.concat(sig,axis=1); ranks=wide.rank(axis=1,pct=True); print('coverage',round(wide.notna().mean().mean(),4),'turnover',round(ranks.diff().abs().mean(axis=1).mean(),4),'period',wide.index.min().date(),wide.index.max().date())
# save signal artifact
wide.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20261022_range_shock_signal.csv',index=False)
