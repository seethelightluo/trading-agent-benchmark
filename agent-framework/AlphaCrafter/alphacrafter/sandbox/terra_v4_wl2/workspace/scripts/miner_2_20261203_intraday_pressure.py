import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date')
 intr=d.close/d.open-1
 # close-to-open pressure, negatively scored for reversal; range-normalized alternative
 rng=(d.high-d.low)/d.close
 f=-(intr.rolling(2,min_periods=2).sum())
 fn=-(intr/(rng.replace(0,np.nan))).rolling(2,min_periods=2).mean()
 y=d.close.pct_change().shift(-1)
 rows.append(pd.DataFrame({'date':d.date,'s':s,'raw':f,'norm':fn,'y':y}))
a=pd.concat(rows)
for c in ['raw','norm']:
 out=[]
 for dt,g in a.dropna(subset=[c,'y']).groupby('date'):
  if len(g)>=8: out.append((dt,spearmanr(g[c],g.y).statistic,len(g)))
 v=np.array([x[1] for x in out]); print(c,'dates',len(v),'avg_n',np.mean([x[2] for x in out]),'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1),'hit',(v>0).mean())
 print('regimes',[(yr,round(np.mean([x[1] for x in out if x[0].year==yr]),5)) for yr in range(2020,2027)])
 print('coverage',a[c].notna().mean())
