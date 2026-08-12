import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
dates=D['SPX'].index
R=pd.DataFrame({s:D[s].close.pct_change().reindex(dates) for s in U})
# Medium-term contrarian signal, lagged to avoid lookahead.
F=-R.rolling(20,min_periods=15).sum().shift(1)
Y=pd.DataFrame({s:D[s].close.pct_change().shift(-1).reindex(dates) for s in U})
q=[];ds=[];ns=[]
for dt in dates:
 z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
 if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ds.append(dt);ns.append(len(z))
q=np.array(q);print('dates',len(q),'meanN',round(np.mean(ns),2),'coverage',round(F.notna().sum().sum()/F.size,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
for yr in range(2020,2027):
 x=q[[d.year==yr for d in ds]];print('regime',yr,'n',len(x),'IC',round(x.mean(),6) if len(x) else None,'ICIR',round(x.mean()/x.std(ddof=1),6) if len(x)>1 else None)
print('turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for h in [5,10]:
 YY=pd.DataFrame({s:D[s].close.shift(-h).div(D[s].close).sub(1).reindex(dates) for s in U});a=[]
 for dt in dates:
  z=pd.DataFrame({'f':F.loc[dt],'y':YY.loc[dt]}).dropna()
  if len(z)>=8:a.append(spearmanr(z.f,z.y).statistic)
 print('horizon',h,'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1),6))
