import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base=Path('../persistent/stock_data'); end=pd.Timestamp('2035-08-15'); px={}
for s in U:
 d=pd.read_csv(base/f'{s}.csv',parse_dates=['date']).sort_values('date'); px[s]=d[d.date<=end].set_index('date')
for h in [10,20,40]:
 rows=[]
 for s,d in px.items():
  r=np.log(d.close).diff(); sig=-r.rolling(30,min_periods=30).std(); f=d.close.shift(-h)/d.close-1
  rows.append(pd.DataFrame({'date':d.index,'factor':sig,'fwd':f}).dropna())
 x=pd.concat(rows,ignore_index=True); a=[]; n=[]
 for dt,g in x.groupby('date'):
  if len(g)>=8 and g.factor.nunique()>1:
   v=spearmanr(g.factor,g.fwd).statistic
   if np.isfinite(v): a.append(v); n.append(len(g))
 a=np.array(a); print('horizon',h,'dates',len(a),'avg_instruments',round(np.mean(n),3),'IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)))
