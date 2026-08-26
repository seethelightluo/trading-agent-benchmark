import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}
for s in U:
 d=get_stock_daily_data(s,days=6000)
 if d is None or len(d)<300: d=get_index_daily_data(s,days=6000)
 if d is not None:p[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(p).sort_index(); r=np.log(P).diff()
# Breakout continuation after quiet consolidation: lagged 20d return, scaled by 60d vol,
# multiplied by inverse recent/long volatility, with all inputs lagged one day.
v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std()
f=((P/P.shift(20)-1)/(v60*np.sqrt(20))*(v60/(v20+1e-12)).clip(0.5,2.0)).shift(1)
Y=P.shift(-10)/P-1
rows=[]; sig=[]
for dt in f.index:
 x=f.loc[dt];y=Y.loc[dt];ok=x.notna()&y.notna()
 if ok.sum()>=8:rows.append((dt,x[ok].corr(y[ok],method='spearman'),ok.sum()));sig.append((dt,*x.reindex(U).tolist()))
I=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z=I.ic
print('factor compression_breakout dates',len(I),'avgN',I.n.mean(),'coverage',I.n.mean()/15)
print('IC10',z.mean(),'ICIR',z.mean()/z.std(),'hit',(z>0).mean(),'recent365',z.tail(365).mean()/z.tail(365).std(),'recent750',z.tail(750).mean()/z.tail(750).std())
for h in [1,5,10,20]:
 Yh=P.shift(-h)/P-1;a=[]
 for dt in f.index:
  x=f.loc[dt];y=Yh.loc[dt];ok=x.notna()&y.notna()
  if ok.sum()>=8:a.append(x[ok].corr(y[ok],method='spearman'))
 print('decay',h,np.nanmean(a))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
I.to_csv('scripts/miner_2_20350201_compression_breakout_ic.csv');pd.DataFrame(sig,columns=['date']+U).set_index('date').to_csv('scripts/miner_2_20350201_compression_breakout_signal.csv')
