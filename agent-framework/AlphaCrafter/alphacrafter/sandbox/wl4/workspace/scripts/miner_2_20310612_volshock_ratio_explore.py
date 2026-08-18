import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2031-06-11')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.loc[:cut] for s in U}
dates=sorted(set.intersection(*[set(x.index) for x in D.values()])); P=pd.DataFrame({s:D[s].reindex(dates) for s in U}); r=P.pct_change()
# lagged relative volatility shock: negative signal favors assets whose short volatility recently spiked
v5=r.rolling(5).std(); v20=r.rolling(20).std(); ratio=(v5/(v20+1e-12)).replace([np.inf,-np.inf],np.nan)
F={'raw':-ratio,'log':-np.log(ratio),'demean':-(ratio-ratio.median(axis=1).values[:,None]),'scaled':-(ratio-1)*r.rolling(20).std()}
def calc(f,H,st=0):
 a=[]; ns=[]
 for i in range(max(st,20),len(P)-H-1):
  x=f.iloc[i].shift(0); y=P.iloc[i+1+H]/P.iloc[i+1]-1; ok=x.notna()&y.notna(); ns.append(ok.sum())
  if ok.sum()>=8 and x[ok].nunique()>1: 
   q=spearmanr(x[ok],y[ok]).statistic
   if np.isfinite(q):a.append(q)
 a=np.array(a); return len(a),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean(),np.mean(ns)
print('dates',len(P),'assets',len(U))
for n,f in F.items():
 for H in (5,10,20): print(n,H,calc(f,H))
 print('recent10',calc(f,10,len(P)-1095),'recent20',calc(f,20,len(P)-1095))
