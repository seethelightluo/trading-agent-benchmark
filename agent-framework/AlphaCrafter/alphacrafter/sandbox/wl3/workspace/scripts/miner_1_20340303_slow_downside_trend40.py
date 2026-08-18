import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index().ffill(); lp=np.log(P); r=lp.diff(); cs=r.sub(r.mean(axis=1),axis=0)
down=r.where(r<0,0).rolling(45,min_periods=20).std()
# slower residual trend, volatility-normalized and gated by lagged cross-sectional dispersion
raw=cs.rolling(40,min_periods=25).mean()/(down+1e-8)
disp=cs.std(axis=1).rolling(90,min_periods=45).rank(pct=True).shift(1)
f=raw.mul((1.0-0.4*disp),axis=0).shift(1)
rows=[]
for h in [10,20]:
 future=lp.shift(-h)-lp; rows=[]
 for dt in f.index:
  a,b=f.loc[dt],future.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8 and a[ok].nunique()>1: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic.dropna()
 print('horizon',h,'dates',len(z),'avgN',z.n.mean(),'coverage',z.n.mean()/15,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turn',f.rank(pct=True).diff().abs().mean(axis=1).mean())
 for n in [120,252,756,1260]:
  x=q.tail(n); print('recent',n,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
 if h==10: z.to_csv('scripts/miner_1_20340303_slow_downside_trend40_ic.csv')
f.to_csv('scripts/miner_1_20340303_slow_downside_trend40_signal.csv')
