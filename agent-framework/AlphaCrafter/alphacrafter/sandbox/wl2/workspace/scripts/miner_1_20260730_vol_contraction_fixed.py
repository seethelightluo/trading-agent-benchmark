import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
def load(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.loc[:cut]
P=pd.concat({a:load(a) for a in U},axis=1).sort_index(); R=P.pct_change()
# Volatility contraction: recent realized risk falling versus medium-term risk; computed from completed returns.
rv10=R.rolling(10,min_periods=8).std(); rv60=R.rolling(60,min_periods=40).std(); F=-(rv10/rv60-1.0)
Y=R.shift(-1)
for h in [1,5,10]:
 y=R.rolling(h,min_periods=h).sum().shift(-h+1); vals=[]; ns=[]; dates=[]
 for d in F.index:
  z=pd.concat([F.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));dates.append(d)
 x=np.array(vals); print('horizon',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()))
print('coverage %.4f turnover %.4f'%(F.stack().notna().mean(),F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for yr in range(2020,2027):
 x=[]
 for d in F.index[(F.index.year==yr)]:
  z=pd.concat([F.loc[d],Y.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('regime',yr,'dates',len(x),'IC',round(np.mean(x),5) if x else None,'ICIR',round(np.mean(x)/np.std(x,ddof=1),4) if len(x)>1 else None)
print('signal_correlations',F.stack().corr((-R.rolling(5).sum()).stack()),F.stack().corr((R.rolling(20).sum()/R.rolling(20).std()).stack()))
