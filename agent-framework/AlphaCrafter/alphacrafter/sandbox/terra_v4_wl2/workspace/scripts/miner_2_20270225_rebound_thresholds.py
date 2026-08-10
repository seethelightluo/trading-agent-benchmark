import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.sort_index() for a in A}
for th in [.03,.05,.07,.10]:
 for h in [1,3,5]:
  sig={a:(-p[a].pct_change(5)).where(p[a]/p[a].rolling(20,min_periods=20).max()-1<=-th) for a in A}; rows=[]
  dates=sorted(set().union(*[x.index for x in sig.values()]))
  for d in dates:
   z=[]
   for a in A:
    if d in sig[a].index:z.append((sig[a].loc[d],p[a].pct_change(h).shift(-h).loc[d]))
   z=pd.DataFrame(z,columns=['x','y']).dropna()
   if len(z)>=8 and z.x.nunique()>1 and z.y.nunique()>1:
    q=spearmanr(z.x,z.y).statistic
    if np.isfinite(q):rows.append(q)
  x=np.array(rows);print('th',th,'h',h,'dates',len(x),'avgN',sum(1 for _ in []) if False else '', 'IC',x.mean() if len(x) else np.nan,'ICIR',x.mean()/x.std(ddof=1) if len(x)>1 else np.nan,'hit',(x>0).mean() if len(x) else np.nan)
# selected 5% 1d artifact
th=.05; sig={a:(-p[a].pct_change(5)).where(p[a]/p[a].rolling(20,min_periods=20).max()-1<=-th) for a in A}
out=pd.DataFrame([(d,a,sig[a].get(d,np.nan)) for d in sorted(set().union(*[x.index for x in sig.values()])) for a in A],columns=['date','symbol','signal']);out.to_csv('../persistent/factor_signals_miner_2_20270225_conditional_rebound5.csv',index=False)
