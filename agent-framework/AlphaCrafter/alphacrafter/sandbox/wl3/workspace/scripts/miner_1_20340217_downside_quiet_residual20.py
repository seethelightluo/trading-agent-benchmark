import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index().ffill(); lp=np.log(P); r=lp.diff()
cs=r.sub(r.mean(axis=1),axis=0)
vol=r.rolling(30,min_periods=15).std()
# Residual trend is penalized by downside deviation, with a lagged quiet-regime gate.
down=r.where(r<0,0).rolling(30,min_periods=15).std()
raw=cs.rolling(20,min_periods=15).mean()/(down+1e-8)
disp=cs.std(axis=1).rolling(60,min_periods=30).rank(pct=True).shift(1)
f=raw.mul((1.0-0.5*disp),axis=0).shift(1)
future=lp.shift(-10)-lp; rows=[]
for dt in f.index:
 a,b=f.loc[dt],future.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8 and a[ok].nunique()>1: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic.dropna()
print('dates',len(z),'avgN',z.n.mean(),'coverage',z.n.mean()/15,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turn',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for n in [120,252,756,1260]:
 x=q.tail(n); print('recent',n,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
for h in [1,5,10,20]:
 y=lp.shift(-h)-lp; rr=[]
 for dt in f.index:
  a,b=f.loc[dt],y.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8 and a[ok].nunique()>1: rr.append(a[ok].corr(b[ok]))
 x=pd.Series(rr).dropna(); print('horizon',h,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'obs',len(x))
f.to_csv('scripts/miner_1_20340217_downside_quiet_residual20_signal.csv'); z.to_csv('scripts/miner_1_20340217_downside_quiet_residual20_ic.csv')
