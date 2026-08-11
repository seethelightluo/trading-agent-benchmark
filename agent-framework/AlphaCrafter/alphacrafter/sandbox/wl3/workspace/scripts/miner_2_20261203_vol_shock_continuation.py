import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT='2026-12-02'
def load(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().query('date<=@CUT')
D={s:load(s) for s in U}
# Volatility shock continuation: recent 5d return weighted by expansion/contraction of short vol versus long vol.
def factor(x):
 r=x.close.pct_change(); v5=r.rolling(5,min_periods=5).std(); v30=r.rolling(30,min_periods=20).std()
 shock=(v5/v30.replace(0,np.nan)).clip(0.25,4)
 return x.close.pct_change(5)*shock
rows=[]
for s,x in D.items():
 f=factor(x); rows.append(pd.DataFrame({'date':x.index,'f':f.to_numpy(),'y':x.close.shift(-1).div(x.close).sub(1).to_numpy()}))
a=pd.concat(rows).dropna(); out=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:out.append((dt,spearmanr(g.f,g.y).statistic,len(g)))
z=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); alln=sum(len(x) for x in D.values()); valid=sum(factor(x).notna().sum() for x in D.values())
r=pd.concat([factor(x).rename(s) for s,x in D.items()],axis=1).rank(axis=1,pct=True)
print('idea vol_shock_continuation cutoff',CUT,'dates',len(z),'avg_n',z.n.mean(),'coverage',valid/alln,'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'hit',(z.ic>0).mean(),'turnover',r.diff().abs().mean(axis=1).mean())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-02')]:
 q=z.loc[lo:hi].ic;print('regime',lo,hi,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
for h in [3,5,10]:
 b=pd.concat([pd.DataFrame({'date':x.index,'f':factor(x).to_numpy(),'y':x.close.shift(-h).div(x.close).sub(1).to_numpy()}) for x in D.values()]).dropna(); q=[]
 for dt,g in b.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:q.append(spearmanr(g.f,g.y).statistic)
 q=pd.Series(q);print('decay',h,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
sig=pd.concat([factor(x).rename(s) for s,x in D.items()],axis=1);sig.index.name='date';sig.to_csv('scripts/miner_2_20261203_vol_shock_continuation_signal.csv')
