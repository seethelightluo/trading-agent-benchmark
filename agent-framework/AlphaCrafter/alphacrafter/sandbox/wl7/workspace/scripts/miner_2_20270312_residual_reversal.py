import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   d=f(s,days=800)
   if d is not None and len(d): return d
  except: pass
arr={}
for s in U:
 d=fetch(s)
 if d is not None: arr[s]=pd.Series(pd.to_numeric(d.close,errors='coerce').values,index=pd.to_datetime(d.date)).sort_index()
p=pd.DataFrame(arr).sort_index(); ret=p.pct_change(); dates=[]
# factor is lagged cross-sectional residual: yesterday's return less cross-sectional median, with volatility scaling
f=-(ret.sub(ret.median(axis=1),axis=0)).shift(1)/ret.rolling(20).std().shift(1)
y=ret.shift(-1); ics=[]; ns=[]
for dt in f.index:
 z=pd.DataFrame({'f':f.loc[dt],'y':y.loc[dt]}).dropna()
 if len(z)>=8: ics.append(z.f.corr(z.y,method='spearman'));ns.append(len(z))
a=pd.Series(ics).dropna(); print('dates',len(a),'avg_n',np.mean(ns),'symbols',len(p.columns),'coverage',f.notna().sum().sum()/(len(f)*len(p.columns)));print('IC %.8f ICIR %.8f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()));print('turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean());print('period',f.index.min(),f.index.max());
f.stack().rename('factor').to_csv('scripts/miner_2_20270312_residual_reversal_signal.csv')
for label,g in [('early',a.iloc[:len(a)//2]),('late',a.iloc[len(a)//2:])]:print(label,len(g),g.mean(),g.mean()/g.std(ddof=1))
