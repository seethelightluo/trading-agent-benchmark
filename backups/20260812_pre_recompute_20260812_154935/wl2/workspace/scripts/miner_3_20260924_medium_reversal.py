import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
end=min(max(x.index.max() for x in D.values()),pd.Timestamp('2026-09-23')); dates=D['SPX'].index[(D['SPX'].index>='2020-04-01')&(D['SPX'].index<=end)]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); r=C.pct_change()
def run(L):
 F=-(C/C.shift(L)-1).shift(1); q=[];ns=[];ds=[]
 for dt in dates:
  z=pd.concat([F.loc[dt].rename('f'),r.shift(-1).loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(dt)
 q=np.array(q); return q,ns,ds,F
for L in [10,20,30,60]:
 q,n,d,F=run(L); ic=q.mean(); print('L',L,'dates',len(q),'avgN',round(np.mean(n),2),'IC',round(ic,6),'ICIR',round(ic/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'cov',round(F.notna().sum().sum()/F.size,4),'turn',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
 for k in [63,126,252]:
  x=q[-k:];print(' recent',k,round(x.mean(),6),round(x.mean()/x.std(ddof=1),6))
print('universe',len(U),'end',end.date())
print('signal_artifact','formula=-(close/close.shift(L)-1), lag=1; L tested 10,20,30,60')
