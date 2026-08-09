import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
def load(path):
 d=pd.read_csv(path,parse_dates=['date']).set_index('date'); return d['close'].pct_change()
px=pd.concat({a:load('../persistent/stock_data/'+a+'.csv') for a in assets},axis=1)
macro=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close'].pct_change()
rets=px; fwd=rets.shift(-1)
for win in [20,40,60,90]:
 # beta to DXY, only completed through t
 cov=rets.rolling(win,min_periods=max(12,win//2)).cov(macro)
 var=macro.rolling(win,min_periods=max(12,win//2)).var()
 beta=cov.div(var,axis=0)
 fac=-beta
 vals=[]; turnovers=[]
 for dt in fac.index:
  x=fac.loc[dt].dropna(); y=fwd.loc[dt].reindex(x.index).dropna(); x=x.reindex(y.index)
  if len(x)>=8 and x.nunique()>1 and y.nunique()>1:
   vals.append(spearmanr(x,y).statistic)
   ranks=x.rank(pct=True); turnovers.append((ranks-ranks.shift(1) if False else 0))
 s=pd.Series(vals)
 print('WIN',win,'dates',len(s),'avgN',len(assets),'IC %.5f ICIR %.5f hit %.3f'%(s.mean(),s.mean()/s.std(),(s>0).mean()))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  z=s[(s.index.year>=lo)&(s.index.year<=hi)] if isinstance(s.index,pd.DatetimeIndex) else s
  # vals lost dates; skip
 # coverage
 print('coverage',fac.notna().sum(axis=1).mean()/len(assets))
 # horizons
 for h in [5,10]:
  yy=rets.shift(-h)
  q=[]
  for dt in fac.index:
   x=fac.loc[dt].dropna(); y=yy.loc[dt].reindex(x.index).dropna(); x=x.reindex(y.index)
   if len(x)>=8 and x.nunique()>1 and y.nunique()>1:q.append(spearmanr(x,y).statistic)
  q=pd.Series(q);print(' h',h,'n',len(q),'IC %.5f IR %.5f'%(q.mean(),q.mean()/q.std()))
