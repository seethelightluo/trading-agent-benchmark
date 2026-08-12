import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
dates=D['SPX'].index; R=pd.DataFrame({s:D[s].close.pct_change().reindex(dates) for s in U}); V=pd.DataFrame({s:D[s].volume.replace(0,np.nan).reindex(dates) for s in U})
# Smooth volume-confirmed reversal: reversal strength is attenuated when recent moves occur on unusually high volume.
vs=np.log(V).sub(np.log(V).rolling(30,min_periods=15).median())
F=(-(R* (1+vs.clip(-2,2))).rolling(3,min_periods=3).sum()).shift(1)
Y=pd.DataFrame({s:D[s].close.shift(-1).div(D[s].close).sub(1).reindex(dates) for s in U});q=[];ns=[];ds=[]
for dt in dates:
 z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
 if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(dt)
q=np.asarray(q);print('dates',len(q),'meanN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
for yr in range(2020,2027):
 x=np.array([q[i] for i,d in enumerate(ds) if d.year==yr]); print('regime',yr,'dates',len(x),'IC',round(x.mean(),6) if len(x) else None,'ICIR',round(x.mean()/x.std(ddof=1),4) if len(x)>1 else None)
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
