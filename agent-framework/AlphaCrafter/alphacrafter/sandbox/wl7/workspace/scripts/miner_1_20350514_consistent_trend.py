import pandas as pd, numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}
px=pd.DataFrame(D).sort_index().loc['2020-01-01':'2035-05-13']; r=px.pct_change()
# Consistent trend: signed 20d return, weighted by fraction of positive daily observations, normalized by trailing volatility.
ret20=px.pct_change(20); vol40=r.rolling(40,min_periods=25).std(); breadth=r.gt(0).rolling(20,min_periods=15).mean()
s=(ret20/(vol40*np.sqrt(20)+1e-12))*(2*breadth-1)
s=s.shift(1)
fwd={h:px.shift(-h)/px-1 for h in [1,5,10,20]}
def ev(x,h):
 z=[]; ns=[]
 for d in x.index:
  q=pd.concat([x.loc[d],fwd[h].loc[d]],axis=1).dropna()
  if len(q)>=8:
   c=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
   if np.isfinite(c):z.append(c); ns.append(len(q))
 a=np.array(z); return len(a),np.mean(ns),a.mean(),a.mean()/(a.std(ddof=1)+1e-12),(a>0).mean()
print('assets',len(A),'dates',len(px),'end',px.index[-1].date())
print('coverage',s.notna().mean().mean(),'turnover',s.rank(pct=True,axis=1).diff().abs().stack().mean())
for h in [1,5,10,20]: print('H',h,ev(s,h))
for start,end in [('2020','2024'),('2025','2029'),('2030','2034'),('2035-01-01','2035-05-13')]:
 x=s.loc[start:end]; print('regime',start,end,'n',len(x),ev(x,10))
for n in [180,500,750]: print('recent',n,ev(s.iloc[-n:],10))
s.to_csv('scripts/miner_1_20350514_consistent_trend_signal.csv')
