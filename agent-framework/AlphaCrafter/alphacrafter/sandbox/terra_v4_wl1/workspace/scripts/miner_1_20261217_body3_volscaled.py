import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy()
 # lagged 3-day signed body pressure, scaled by prior 20d volatility; reversal interpretation
 body=(d.close-d.open)/d.open
 vol=d.close.pct_change().rolling(20,min_periods=15).std()
 d['f']=-(body.rolling(3,min_periods=3).mean().shift(1))/(vol.shift(1)+1e-12)
 for h in [1,5,10]: d['y'+str(h)]=d.close.shift(-h)/d.close-1
 rows.append(d[['date','f','y1','y5','y10']].assign(symbol=s))
x=pd.concat(rows)
for h in [1,5,10]:
 a=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['f','y'+str(h)])
  if len(g)>=8:
   c=spearmanr(g.f,g['y'+str(h)]).statistic
   if np.isfinite(c): a.append((dt,c,len(g)))
 z=pd.DataFrame(a,columns=['date','ic','n']).set_index('date'); q=z.ic
 print('H',h,'dates',len(q),'avgN',round(z.n.mean(),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if h==1:
  for reg,g in q.groupby(pd.cut(q.index.year,[2019,2022,2024,2026,2027])): print('REG',reg,'n',len(g),'IC',round(g.mean(),6),'ICIR',round(g.mean()/g.std(ddof=1),6))
f=x.dropna(subset=['f']); r=f.pivot(index='date',columns='symbol',values='f').rank(axis=1,pct=True)
print('coverage',round(len(f)/len(x),4),'turnover',round(r.diff().abs().mean(axis=1).mean(),4),'period',x.date.min(),x.date.max())
f.to_csv('scripts/miner_1_20261217_body3_volscaled_signal.csv',index=False)
