import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   d=f(s,days=800)
   if d is not None and len(d): return d
  except: pass
rows=[]
for s in U:
 d=fetch(s)
 if d is None: continue
 d=d.sort_values('date').copy(); c=pd.to_numeric(d.close,errors='coerce'); o=pd.to_numeric(d.open,errors='coerce'); h=pd.to_numeric(d.high,errors='coerce'); l=pd.to_numeric(d.low,errors='coerce')
 # lagged intraday reversal, scaled by range; range is a noise/liquidity adjustment
 intr=(c/o-1); rng=(h/l-1).rolling(20).median()
 sig=-(intr/rng.replace(0,np.nan)).shift(1)
 fw=c.pct_change().shift(-1)
 for dt,v,y in zip(pd.to_datetime(d.date),sig,fw): rows.append((dt,s,v,y))
x=pd.DataFrame(rows,columns=['date','symbol','factor','fwd']).dropna(); out=[]; ns=[]
for dt,g in x.groupby('date'):
 if len(g)>=8: out.append(g.factor.corr(g.fwd,method='spearman')); ns.append(len(g))
a=pd.Series(out).dropna(); print('dates',len(a),'avg_n',np.mean(ns),'symbols',x.symbol.nunique(),'coverage',len(x)/(len(set(z[0] for z in rows))*15)); print('IC %.8f ICIR %.8f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean())); print('turnover',x.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True).diff().abs().mean().mean()); print('period',x.date.min(),x.date.max()); x.to_csv('scripts/miner_2_20270312_intraday_reversal_signal.csv',index=False)
for label,g in [('early',a.iloc[:len(a)//2]),('late',a.iloc[len(a)//2:])]: print(label,len(g),g.mean(),g.mean()/g.std(ddof=1))
