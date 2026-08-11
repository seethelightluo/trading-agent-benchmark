import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
dates=D['SPX'].index; P=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); R=P.pct_change()
# Persistence-weighted 60d trend: cumulative return multiplied by fraction of positive sessions, lagged.
r60=R.rolling(60,min_periods=40).sum(); consistency=(R>0).rolling(60,min_periods=40).mean()
F=(r60*consistency).shift(1); Y=P.shift(-1).div(P).sub(1)
q=[];ns=[];ds=[]
for dt in dates:
 z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
 if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(dt)
q=np.asarray(q);print('dates',len(q),'meanN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
for yr in range(2020,2027):
 x=q[[d.year==yr for d in ds]];print('regime',yr,len(x),round(x.mean(),6) if len(x) else None,round(x.mean()/x.std(ddof=1),6) if len(x)>1 else None)
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for h in [5,10]:
 Yh=P.shift(-h).div(P).sub(1);a=[]
 for dt in dates:
  z=pd.DataFrame({'f':F.loc[dt],'y':Yh.loc[dt]}).dropna()
  if len(z)>=8:a.append(spearmanr(z.f,z.y).statistic)
 print('horizon',h,'dates',len(a),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1),6))
