import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end='2026-07-15'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}
dates=D['SPX'].index
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U},index=dates)
H=pd.DataFrame({s:D[s].high.reindex(dates) for s in U},index=dates); L=pd.DataFrame({s:D[s].low.reindex(dates) for s in U},index=dates)
# Blend two interpretable lagged reversal components: close near low and recovery from 60d low.
clv=-(2*(C-L)/(H-L)).shift(1)
rec=(C/C.rolling(60,min_periods=40).min()-1).shift(1)
def pct(x): return x.rank(axis=1,pct=True)
F=pct(clv)+0.25*pct(rec)
print('idea clv_recovery_blend universe',len(U),'dates',len(dates))
for h in [1,5,10]:
 Y=C.shift(-h).div(C)-1; q=[]; ns=[]; ds=[]
 for dt in dates:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   r=spearmanr(z.f,z.y).statistic
   if np.isfinite(r): q.append(r);ns.append(len(z));ds.append(dt)
 q=np.array(q); print('horizon',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if h==1:
  for yr in range(2020,2027):
   x=q[[d.year==yr for d in ds]];print('regime',yr,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),4))
  for k in [252,504]:
   x=q[-k:];print('recent',k,round(x.mean(),6),round(x.mean()/x.std(ddof=1),6))
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
print('corr_clv',round(pct(clv).stack().corr(pct(rec).stack(),method='spearman'),4))
print('corr_mom20',round(pct(rec).stack().corr(pct(C.pct_change().rolling(20).sum()).stack(),method='spearman'),4))
