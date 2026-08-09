import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).sort_values('date')
v['vz']=(v.close-v.close.rolling(60,min_periods=20).mean())/(v.close.rolling(60,min_periods=20).std()+1e-9)
v['vz']=v.vz.clip(-2,2)
rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').merge(v[['date','vz']],on='date',how='left')
 d['factor']=-d.close.pct_change(3)*(1+d.vz.clip(lower=-.5))
 d['f1']=d.close.shift(-1)/d.close-1; d['f5']=d.close.shift(-5)/d.close-1; d['f10']=d.close.shift(-10)/d.close-1
 rows.append(d[['date','factor','f1','f5','f10']].assign(symbol=s))
a=pd.concat(rows)
def calc(c):
 q=[]
 for dt,g in a.dropna(subset=['factor',c]).groupby('date'):
  if len(g)>=8 and g.factor.nunique()>1:q.append(spearmanr(g.factor,g[c]).statistic)
 q=np.array(q); return len(q),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()
print('VIX-conditioned continuous 3d reversal')
for c in ['f1','f5','f10']: print(c,calc(c))
print('coverage',a.factor.notna().mean(),'avg names',a.dropna(subset=['factor']).groupby('date').size().mean())
a.pivot(index='date',columns='symbol',values='factor').to_csv('scripts/miner_1_20270128_vix_reversal_signal.csv')
