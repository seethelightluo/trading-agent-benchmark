import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index().ffill(); lp=np.log(P); r=lp.diff()
# Drawdown-recovery factor: favor assets that have recovered recently from a deep
# medium-term drawdown, normalized by their own volatility. All inputs lagged one day.
peak=lp.rolling(120,min_periods=60).max()
dd=lp-peak
recovery=lp.diff(10)-lp.diff(40) # recent rebound relative to medium-term trend
vol=r.rolling(30,min_periods=20).std()
f=(recovery/(vol+1e-8)).rank(axis=1,pct=True).sub(.5,axis=0).shift(1)
rows=[]; future={h:lp.shift(-h)-lp for h in [1,5,10,20]}
for dt in f.index:
 a=f.loc[dt]
 for h,y in future.items():
  b=y.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8 and a[ok].nunique()>1: rows.append((dt,h,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10,20]:
 q=z[z.h==h].ic.dropna(); print('horizon',h,'dates',len(q),'avgN',z[z.h==h].n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
q=z[z.h==10].set_index('date').ic.dropna()
for n in [120,252,756,1260]:
 x=q.tail(n); print('recent',n,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
print('turn',f.rank(pct=True).diff().abs().mean(axis=1).mean(),'coverage',f.notna().mean().mean(),'dates',len(q))
f.to_csv('scripts/miner_3_20340303_drawdown_recovery_signal.csv'); z.to_csv('scripts/miner_3_20340303_drawdown_recovery_ic.csv')
