import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index().ffill(); lp=np.log(P); r=lp.diff()
# Volatility-transition trend: 20d residual momentum favored when volatility is contracting vs its 60d baseline.
res20=(lp-lp.shift(20)).sub((lp-lp.shift(20)).mean(axis=1),axis=0)
vol20=r.rolling(20,min_periods=15).std(); vol60=r.rolling(60,min_periods=45).std()
state=(1-(vol20/vol60).clip(0.25,2.0)/2.0).rank(axis=1,pct=True)
f=(res20*state).shift(1)
future=lp.shift(-10)-lp; rows=[]
for dt in f.index:
 a,b=f.loc[dt],future.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8 and a[ok].nunique()>1: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic.dropna()
print('factor volatility_contraction_residual_momentum20 dates',len(z),'avgN',round(z.n.mean(),3),'coverage',round(z.n.mean()/15,5),'IC',round(q.mean(),7),'ICIR',round(q.mean()/q.std(ddof=1),7),'hit',round((q>0).mean(),5),'turn',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),5))
for n in [120,252,756,1260]:
 x=q.tail(n); print('recent',n,'IC',round(x.mean(),7),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),5))
for h in [1,5,10,20]:
 yy=lp.shift(-h)-lp; rr=[]
 for dt in f.index:
  a,b=f.loc[dt],yy.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8 and a[ok].nunique()>1: rr.append(a[ok].corr(b[ok]))
 x=pd.Series(rr).dropna(); print('horizon',h,'IC',round(x.mean(),7),'ICIR',round(x.mean()/x.std(ddof=1),6),'obs',len(x))
f.to_csv('scripts/miner_2_20340303_volatility_contraction_momentum20_signal.csv'); z.to_csv('scripts/miner_2_20340303_volatility_contraction_momentum20_ic.csv')
