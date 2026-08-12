import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
# Intraday pressure: fading a completed session's open-to-close move.
F=pd.DataFrame({s:-(D[s].close/D[s].open-1).shift(1) for s in U}).sort_index()
for h in [1,5,10]:
 Y=pd.DataFrame({s:D[s].close.shift(-h)/D[s].close-1 for s in U}).sort_index(); q=[];ns=[]
 for dt in F.index:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
 q=np.array(q); print('horizon',h,'dates',len(q),'meanN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if h==1:
  for yr in range(2020,2027):
   a=[]
   for dt in F.loc[str(yr)].index:
    z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
    if len(z)>=8:a.append(spearmanr(z.f,z.y).statistic)
   print(yr,len(a),round(np.mean(a),6) if a else None,round(np.mean(a)/np.std(a,ddof=1),4) if len(a)>1 else None)
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
