import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2031-01-08')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.loc[:cut] for s in U}
dates=sorted(set.intersection(*[set(x.index) for x in px.values()])); P=pd.DataFrame({s:px[s].reindex(dates) for s in U}); r=P.pct_change()
# Recovery strength: recent 20d gain relative to trailing 60d peak-to-trough drawdown; lag one day.
rollmax=P.rolling(60,min_periods=60).max(); dd=(P/rollmax-1).rolling(60,min_periods=60).min().abs()
f=(r.rolling(20,min_periods=20).sum()/dd.replace(0,np.nan)).shift(1)
def run(H,start=0):
 a=[]; ns=[]
 for i in range(start,len(P)-H-1):
  x=f.iloc[i]; y=P.iloc[i+1+H]/P.iloc[i+1]-1; ok=x.notna()&y.notna(); n=int(ok.sum()); ns.append(n)
  if n>=8 and x[ok].nunique()>1 and y[ok].nunique()>1:
   z=spearmanr(x[ok],y[ok]).statistic
   if np.isfinite(z): a.append(z)
 a=np.asarray(a); return len(a),float(a.mean()),float(a.mean()/a.std(ddof=1)),float((a>0).mean()),float(np.mean(ns)),float(np.mean(f.notna().mean(axis=1)))
print('dates',len(P),'assets',len(U))
for H in [1,5,10,20]: print('H',H,run(H))
for n in [365,730,1095]: print('recent',n,'h10',run(10,max(0,len(P)-n-11)))
rank=f.rank(axis=1,pct=True); print('rank-turnover',float((rank.diff().abs().mean(axis=1)>0.10).mean()))
