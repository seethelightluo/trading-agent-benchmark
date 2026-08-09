import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Macro-conditioned defensive rotation: lagged 10-session relative strength, sign reversed in stressed VIX regime.
macro=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).sort_values('date'); macro=macro[macro.date<=END]; vc='close' if 'close' in macro else 'value'; macro['vixchg']=macro[vc].pct_change(20).shift(1); macro['stress']=(macro[vc].shift(1)>macro[vc].shift(1).rolling(252,min_periods=60).quantile(.7)).astype(float)
macro=macro[['date','vixchg','stress']]
rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy(); d['ret10']=d.close.shift(1).pct_change(10)
 d=d.merge(macro,on='date',how='left'); d['factor']=d.ret10*(1-2*d.stress) # momentum calm, reversal stress
 for h in [1,5,10]: d[f'y{h}']=d.close.shift(-h)/d.close-1
 rows.append(d[['date','factor','y1','y5','y10']].assign(symbol=s))
x=pd.concat(rows,ignore_index=True)
for h in [1,5,10]:
 a=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',f'y{h}'])
  if len(g)>=8 and g.factor.nunique()>1 and g[f'y{h}'].nunique()>1:
   r=spearmanr(g.factor,g[f'y{h}']).statistic
   if np.isfinite(r): a.append((dt,r,len(g)))
 z=pd.DataFrame(a,columns=['date','ic','n']); q=z.ic
 print('H',h,'dates',len(q),'avgN',round(z.n.mean(),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if h==1:
  for yr,g in z.groupby(z.date.dt.year): print('YR',yr,len(g),round(g.ic.mean(),5),round(g.ic.mean()/g.ic.std(ddof=1),4))
v=x.dropna(subset=['factor']); print('coverage',round(len(v)/len(x),4),'symbols',x.symbol.nunique(),'period',x.date.min(),x.date.max())
x.to_csv('scripts/miner_2_20261217_macro_defensive_rotation_signal.csv',index=False)
