import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for s in U}).sort_index().ffill(); L=np.log(P); R=L.diff()
# Multi-horizon trend agreement: relative 60-session return, weighted by agreement of 10/30/60 direction and inverse realized risk.
rel=L.diff(60).sub(L.diff(60).median(axis=1),axis=0)
agree=(np.sign(L.diff(10))+np.sign(L.diff(30))+np.sign(L.diff(60)))/3
vol=R.rolling(30,min_periods=20).std()
f=rel*agree/(vol+1e-8); f=f.shift(1)
for h in [1,5,10,20]:
 y=L.shift(-h)-L; rows=[]
 for dt in f.index:
  a,b=f.loc[dt],y.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic
 print('horizon',h,'dates',len(z),'avgN',round(z.n.mean(),2),'coverage',round(z.n.mean()/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'turn',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),4))
 for n in [120,252,756,1260]:
  x=q.tail(n); print('recent',n,'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
 if h==10: z.to_csv('scripts/miner_2_20330930_trend_agreement60_ic.csv')
f.to_csv('scripts/miner_2_20330930_trend_agreement60_signal.csv')
