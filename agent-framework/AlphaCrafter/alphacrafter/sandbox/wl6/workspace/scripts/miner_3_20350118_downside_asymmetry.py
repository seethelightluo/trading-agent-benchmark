import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100: d=get_index_daily_data(s,5000)
 return None if d is None else d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.concat({s:load(s) for s in U if load(s) is not None},axis=1).sort_index()
r=np.log(P/P.shift(1)); neg=r.where(r<0,0.0)
# Downside asymmetry: share of 20d squared variation contributed by negative returns.
# Lag one day, high values indicate negatively skewed recent tape.
down_var=(neg**2).rolling(20).sum(); total_var=(r**2).rolling(20).sum()
f=(down_var/total_var).shift(1)
rows=[]
for h in [5,10,20,40]:
 fw=P.shift(-h)/P-1; ics=[]; ns=[]; turns=[]; prev=None
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c):
    ics.append(c);ns.append(len(z)); sig=z.iloc[:,0].rank(pct=True)
    if prev is not None: turns.append(np.mean(abs(sig-prev)))
    prev=sig
 a=np.array(ics)
 print('H',h,'dates',len(a),'avg_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/len(U),4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(len(a)),3),'hit',round(np.mean(a>0),4),'turn',round(np.mean(turns),4))
out=[]
for dt in f.index:
 for s in f.columns:
  if pd.notna(f.loc[dt,s]): out.append([str(dt.date()),s,float(f.loc[dt,s])])
pd.DataFrame(out,columns=['date','symbol','signal']).to_csv('scripts/miner_3_20350118_downside_asymmetry_signal.csv',index=False)
