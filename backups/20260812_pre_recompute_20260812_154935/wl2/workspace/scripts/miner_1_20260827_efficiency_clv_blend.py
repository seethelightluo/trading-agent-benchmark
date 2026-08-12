import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
dates=D['SPX'].index; P=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); R=P.pct_change()
# Efficiency-CLV blend: directional efficiency over 20 sessions times recent close-location pressure.
e20=R.rolling(20,min_periods=15).sum().div(R.abs().rolling(20,min_periods=15).sum()+1e-12)
clv=pd.DataFrame({s:((D[s].close-D[s].low)/(D[s].high-D[s].low+1e-12)).reindex(dates) for s in U}).rolling(5,min_periods=3).mean()
# rank blend, lagged to avoid lookahead
F=(e20.rank(axis=1,pct=True)+clv.rank(axis=1,pct=True)).shift(1)
Y=P.shift(-1).div(P).sub(1)
q=[]; ns=[]; ds=[]
for dt in dates:
 z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
 if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(dt)
q=np.array(q); print('idea efficiency20_clv5_rankblend universe',len(U),'dates',len(q),'avgN',round(np.mean(ns),2))
print('IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
for yr in range(2020,2027):
 x=q[[d.year==yr for d in ds]]; print('regime',yr,len(x),round(x.mean(),6) if len(x) else None)
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for h in [5,10]:
 Yh=P.shift(-h).div(P).sub(1); a=[]
 for dt in dates:
  z=pd.DataFrame({'f':F.loc[dt],'y':Yh.loc[dt]}).dropna()
  if len(z)>=8:a.append(spearmanr(z.f,z.y).statistic)
 print('horizon',h,'dates',len(a),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1),6))
print('signal_artifact /tmp/efficiency20_clv5_rankblend_signal.csv')
F.to_csv('/tmp/efficiency20_clv5_rankblend_signal.csv')
