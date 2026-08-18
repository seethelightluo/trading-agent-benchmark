import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index().ffill(); lp=np.log(P); r=lp.diff()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index()['close'].astype(float)
vr=np.log(v).diff().reindex(r.index).ffill()
# Macro-shock sensitivity: rolling asset beta to VIX innovations times the latest
# 5-session VIX shock. Negative beta is preferred when volatility is rising.
win=60
cov=r.rolling(win,min_periods=40).cov(vr)
var=vr.rolling(win,min_periods=40).var()
beta=cov.div(var,axis=0)
shock=vr.rolling(5,min_periods=4).sum()
f=(-beta.mul(shock,axis=0)).rank(axis=1,pct=True).sub(.5,axis=0).shift(1)
rows=[]; future={h:lp.shift(-h)-lp for h in [1,3,5,10,20]}
for dt in f.index:
 a=f.loc[dt]
 for h,y in future.items():
  b=y.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8 and a[ok].nunique()>1: rows.append((dt,h,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,3,5,10,20]:
 q=z[z.h==h].ic.dropna(); print('horizon',h,'dates',len(q),'avgN',z[z.h==h].n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
q=z[z.h==10].set_index('date').ic.dropna()
for n in [120,252,756,1260]:
 x=q.tail(n); print('recent',n,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
print('turn',f.rank(pct=True).diff().abs().mean(axis=1).mean(),'coverage',f.notna().mean().mean(),'dates',len(q),'avg instruments',z[z.h==10].n.mean())
f.to_csv('scripts/miner_3_20340512_vix_beta_shock_signal.csv'); z.to_csv('scripts/miner_3_20340512_vix_beta_shock_ic.csv')
