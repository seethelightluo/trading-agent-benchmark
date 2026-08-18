import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P='../persistent/stock_data'; M='../persistent/index_data'
prices=pd.DataFrame({s:pd.read_csv(os.path.join(P,s+'.csv'),parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index()
ret=prices.pct_change()
v=pd.read_csv(os.path.join(M,'VIX.csv'),parse_dates=['date']).set_index('date')['close'].sort_index().pct_change()
stress=v.rolling(5,min_periods=5).sum().clip(lower=-.25,upper=.50).fillna(0)
# Stress-amplified reversal: ordinary 5d reversal, with stronger contrarian score after VIX rises.
f=(-prices.pct_change(5)).mul(1+stress,axis=0).shift(0).replace([np.inf,-np.inf],np.nan)
def ic_at(dt,h):
 if dt not in f.index:return np.nan,0
 try:y=(1+ret.loc[dt:].iloc[1:h+1]).prod()-1
 except:return np.nan,0
 z=pd.concat([f.loc[dt],y],axis=1).dropna()
 if len(z)<8 or z.iloc[:,0].nunique()<2 or z.iloc[:,1].nunique()<2:return np.nan,len(z)
 return spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)
for h in [1,5,10,20]:
 a=[]; ns=[]
 for d in prices.index:
  x,n=ic_at(d,h)
  if np.isfinite(x):a.append(x);ns.append(n)
 a=pd.Series(a); print('H',h,'dates',len(a),'avg_n',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1)*np.sqrt(len(a)),'hit',(a>0).mean())
# daily regime and diagnostics
q=[]; ns=[]
for d in prices.index:
 x,n=ic_at(d,1)
 if np.isfinite(x):q.append((d,x));ns.append(n)
q=pd.DataFrame(q,columns=['date','ic']).set_index('date')
print('coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean(),'daily_dates',len(q),'avg_n',np.mean(ns),'min_n',min(ns))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2028-07-26')]:
 z=q.loc[a:b].ic; print('regime',a,b,len(z),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(len(z)))
z=q.ic.tail(250);print('recent250',z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(len(z)))
