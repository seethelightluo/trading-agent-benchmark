import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date')
 r=d.close.pct_change(3)
 # Convexly emphasize larger prior losses; zero for prior gains
 d['factor']=np.where(r<0,(-r)**1.5,0.0)
 d['f1']=d.close.shift(-1)/d.close-1; d['f5']=d.close.shift(-5)/d.close-1; d['f10']=d.close.shift(-10)/d.close-1
 rows.append(d[['date','factor','f1','f5','f10']].assign(symbol=s))
a=pd.concat(rows)
for c in ['f1','f5','f10']:
 q=[]
 for dt,g in a.dropna(subset=['factor',c]).groupby('date'):
  if len(g)>=8 and g.factor.nunique()>1:q.append(spearmanr(g.factor,g[c]).statistic)
 q=np.array(q); print(c,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
print('coverage',a.factor.notna().mean(),'avg names',a.dropna(subset=['factor']).groupby('date').size().mean())
a.pivot(index='date',columns='symbol',values='factor').to_csv('scripts/miner_1_20270128_convex_downside_reversal_signal.csv')
