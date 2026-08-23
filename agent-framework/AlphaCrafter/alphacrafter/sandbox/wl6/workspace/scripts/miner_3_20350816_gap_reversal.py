import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base=Path('../persistent/stock_data'); end=pd.Timestamp('2035-08-15'); px={}
for s in U:
 d=pd.read_csv(base/f'{s}.csv',parse_dates=['date']).sort_values('date'); px[s]=d[d.date<=end].set_index('date')
def calc(h):
 rows=[]
 for s,d in px.items():
  gap=np.log(d.open/d.close.shift(1)); sig=-gap.rolling(10,min_periods=10).mean(); fr=d.close.shift(-h)/d.close-1
  q=pd.DataFrame({'factor':sig,'fwd':fr}).dropna().reset_index(); q['symbol']=s; rows.append(q)
 x=pd.concat(rows,ignore_index=True); ics=[]; counts=[]
 for dt,g in x.groupby('date'):
  if len(g)>=8 and g.factor.nunique()>1 and g.fwd.nunique()>1:
   v=spearmanr(g.factor,g.fwd).statistic
   if np.isfinite(v): ics.append(v); counts.append(len(g))
 a=np.array(ics); return a,counts,len(x)
for h in [5,10,20,40]:
 a,c,n=calc(h); print('horizon',h,'dates',len(a),'avg_instruments',round(np.mean(c),3),'cells',n,'IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)))
