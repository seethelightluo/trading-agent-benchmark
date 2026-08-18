import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for s in U}).sort_index().ffill(); L=np.log(P); R=L.diff()
macro=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].astype(float).reindex(P.index).ffill()
# Stress-adaptive residual momentum: favor medium-term relative strength in calm regimes,
# but reverse the signal during unusually high volatility stress.
rel=L.diff(30).sub(L.diff(30).median(axis=1),axis=0)
stress=(macro-macro.rolling(252,min_periods=100).mean())/(macro.rolling(252,min_periods=100).std()+1e-8)
state=(1-0.8*np.tanh(stress/1.5))
vol=R.rolling(30,min_periods=20).std()
f=rel.div(vol+1e-8).mul(state,axis=0).shift(1)
for h in [1,5,10,20]:
 y=L.shift(-h)-L; rows=[]
 for dt in f.index:
  a,b=f.loc[dt],y.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic
 print('horizon',h,'dates',len(z),'avgN',round(z.n.mean(),2),'coverage',round(z.n.mean()/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'turn',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),4))
 for n in [120,252,756,1260]:
  x=q.tail(n); print('recent',n,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
 if h==10: z.to_csv('scripts/miner_2_20330930_stress_adaptive_momentum_ic.csv')
f.to_csv('scripts/miner_2_20330930_stress_adaptive_momentum_signal.csv')
