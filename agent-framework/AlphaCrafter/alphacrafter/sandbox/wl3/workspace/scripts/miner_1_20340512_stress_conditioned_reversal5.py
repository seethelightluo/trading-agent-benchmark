import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index().ffill(); lp=np.log(P); r=lp.diff()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].astype(float).reindex(P.index).ffill()
# Stress-conditioned short-horizon reversal: reverse 5-session relative returns only when VIX is above its trailing 60-session median; scale by cross-sectional dispersion.
def z(x):
 m=x.mean(axis=1); sd=x.std(axis=1).replace(0,np.nan); return x.sub(m,axis=0).div(sd,axis=0)
rel=r.rolling(5,min_periods=4).sum(); disp=r.rolling(20,min_periods=15).std().mean(axis=1)
stress=(vix>vix.rolling(60,min_periods=40).median()).astype(float)
f=(-z(rel)*stress.values[:,None]/disp.replace(0,np.nan).values[:,None]).shift(1)
# use rank to limit scaling extremes
f=f.rank(axis=1,pct=True).sub(.5,axis=0)
rows=[]; fut={h:lp.shift(-h)-lp for h in [1,3,5,10,20]}
for dt in f.index:
 a=f.loc[dt]
 for h,y in fut.items():
  b=y.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8 and a[ok].nunique()>1: rows.append((dt,h,a[ok].corr(b[ok]),ok.sum()))
zout=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,3,5,10,20]:
 q=zout[zout.h==h].ic.dropna(); print('horizon',h,'dates',len(q),'avgN',zout[zout.h==h].n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
q=zout[zout.h==10].set_index('date').ic.dropna()
for n in [120,252,756,1260]:
 x=q.tail(n); print('recent',n,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
print('turn',f.rank(pct=True).diff().abs().mean(axis=1).mean(),'coverage',f.notna().mean().mean(),'dates',len(q),'avg instruments',zout[zout.h==10].n.mean())
f.to_csv('scripts/miner_1_20340512_stress_conditioned_reversal5_signal.csv'); zout.to_csv('scripts/miner_1_20340512_stress_conditioned_reversal5_ic.csv')
