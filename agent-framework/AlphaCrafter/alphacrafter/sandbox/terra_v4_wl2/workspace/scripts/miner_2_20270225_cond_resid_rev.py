import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}; r={a:x.pct_change() for a,x in p.items()}; m=r['SPX']; v=m.rolling(60,min_periods=40).var()
base={}
for a in A:
 b=r[a].rolling(60,min_periods=40).cov(m)/(v+1e-12); base[a]=-(p[a]/p[a].shift(5)-1-b*(p['SPX']/p['SPX'].shift(5)-1))
for mode in ['bear','bull','all']:
 rows=[]
 for dt in sorted(set().union(*[set(x.index) for x in p.values()])):
  mr=p['SPX'].pct_change(20).get(dt,np.nan); active=(mode=='all' or (mode=='bear' and mr<0) or (mode=='bull' and mr>=0))
  if not active: continue
  vals={a:base[a].get(dt,np.nan) for a in A}; g=[x for x in vals.values() if np.isfinite(x)]; med=np.nanmedian(g) if len(g)>=8 else np.nan; f=[];y=[]
  for a in A:
   if dt not in p[a].index: continue
   i=p[a].index.get_loc(dt); x=vals[a]-med if np.isfinite(vals[a]) and np.isfinite(med) else np.nan
   if np.isfinite(x) and i+1<len(p[a]): f.append(x);y.append(p[a].iloc[i+1]/p[a].iloc[i]-1)
  if len(f)>=8: rows.append(spearmanr(f,y).statistic)
 z=np.array(rows); print(mode,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
