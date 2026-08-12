import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end='2026-07-15'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}
dates=D['SPX'].index; P=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); R=P.pct_change(); M=R.median(axis=1)
# 20-session rolling market-beta residual of prior 5-session return; all inputs lagged one day.
res=pd.DataFrame(index=dates,columns=U,dtype=float)
for s in U:
    x=R[s]; cov=x.rolling(20,min_periods=15).cov(M); var=M.rolling(20,min_periods=15).var()
    beta=cov/var.replace(0,np.nan)
    res[s]=x-beta*M
F=(-res.rolling(5,min_periods=5).sum()).shift(1)
print('idea beta_neutral_5d_reversal; universe',len(U),'dates',len(dates))
for h in [1,5,10]:
 Y=P.shift(-h).div(P).sub(1); q=[]; ns=[]; ds=[]
 for dt in dates:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(dt)
 q=np.asarray(q); print('horizon',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if h==1:
  for yr in range(2020,2027):
   a=q[[d.year==yr for d in ds]]; print('regime',yr,len(a),round(a.mean(),6) if len(a) else None)
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
