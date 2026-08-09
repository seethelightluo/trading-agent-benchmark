import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy(); r=d.close.pct_change();
 # lagged one-day reversal amplified by abnormal prior volume, with cross-sectional activity only
 sh=(d.volume.shift(1)/(d.volume.shift(2).rolling(20,min_periods=10).median()+1e-12)-1).clip(lower=0)
 d['factor']=-r.shift(1)*np.log1p(sh)
 for h in (1,5,10): d['y'+str(h)]=d.close.shift(-h)/d.close-1
 rows.append(d[['date','factor','y1','y5','y10']].assign(symbol=s))
x=pd.concat(rows,ignore_index=True)
for h in (1,5,10):
 a=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor','y'+str(h)])
  if len(g)>=8 and g.factor.nunique()>1: a.append(spearmanr(g.factor,g['y'+str(h)]).statistic)
 a=np.asarray(a); print('H',h,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
v=x.factor.notna(); print('coverage',round(v.mean(),4)); print('period',x.date.min(),x.date.max(),'symbols',x.symbol.nunique()); x.to_csv('scripts/miner_1_20261217_volume_shock_reversal_signal.csv',index=False)
