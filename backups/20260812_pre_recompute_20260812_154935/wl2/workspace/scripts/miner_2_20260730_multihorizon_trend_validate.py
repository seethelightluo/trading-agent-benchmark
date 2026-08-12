import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end='2026-07-15'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}
dates=D['SPX'].index
R=pd.DataFrame({s:D[s].close.pct_change().reindex(dates) for s in U})
raw=(np.sign(R.rolling(5,min_periods=5).sum())+np.sign(R.rolling(20,min_periods=15).sum())+np.sign(R.rolling(60,min_periods=40).sum()))/3
F=raw.shift(1)
Y={h:pd.DataFrame({s:D[s].close.shift(-h).div(D[s].close).sub(1).reindex(dates) for s in U}) for h in [1,5,10]}
def evaluate(y):
 q=[]; ns=[]; ds=[]
 for dt in dates:
  z=pd.DataFrame({'f':F.loc[dt],'y':y.loc[dt]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   v=spearmanr(z.f,z.y).statistic
   if np.isfinite(v): q.append(v);ns.append(len(z));ds.append(dt)
 a=np.asarray(q); return a,np.asarray(ns),ds
for h in [1,5,10]:
 a,n,_=evaluate(Y[h]); print('horizon',h,'dates',len(a),'avg_names',round(n.mean(),3),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
a,n,ds=evaluate(Y[1]);
for yr in range(2020,2027):
 x=a[[d.year==yr for d in ds]]; print('regime',yr,'dates',len(x),'IC',round(x.mean(),6) if len(x) else None,'ICIR',round(x.mean()/x.std(ddof=1),6) if len(x)>1 else None)
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
F.rename_axis('date').reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_2_20260730_multihorizon_trend_signals.csv',index=False)
