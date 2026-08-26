import pandas as pd, numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}
px=pd.DataFrame(D).sort_index().loc['2020-01-01':'2034-07-23']; r=px.pct_change()
# Trend-consistency momentum: medium-horizon return, penalized by realized volatility and rewarded by directional persistence.
ret=px.pct_change(20); vol=r.rolling(40,min_periods=25).std(); consistency=(np.sign(r).rolling(20,min_periods=15).mean()).abs()
s=(ret/(vol*np.sqrt(20)+1e-12))*consistency
s=s.shift(1)
def ev(h,sub=s):
 f=px.shift(-h)/px-1; z=[]; ns=[]
 for d in sub.index:
  q=pd.concat([sub.loc[d],f.loc[d]],axis=1).dropna()
  if len(q)>=8:
   c=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
   if np.isfinite(c): z.append(c);ns.append(len(q))
 a=np.array(z)
 return len(a),float(np.mean(ns)),float(a.mean()),float(a.mean()/(a.std(ddof=1)+1e-12)),float((a>0).mean())
rank=s.rank(pct=True,axis=1)
turn=float(rank.diff().abs().stack().mean())
print('candidate trend_consistent_momentum20')
print('cutoff',px.index[-1].date(),'dates',len(px),'assets',len(A),'coverage',float(s.notna().mean().mean()),'turnover',turn)
for h in [1,5,10,20]: print('H',h,ev(h))
for n in [180,500,750]: print('recent',n,ev(10,s.iloc[-n:]))
print('avgN',sum(len(pd.concat([s.loc[d],(px.shift(-10)/px).loc[d]],axis=1).dropna())>=8 for d in s.index))
s.to_csv('scripts/miner_1_20340724_trend_consistent_momentum_signal.csv')
