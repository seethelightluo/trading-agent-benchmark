import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index().ffill(); lp=np.log(P); r=lp.diff(); res=r.sub(r.mean(axis=1),axis=0)
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index()['close'].astype(float).reindex(P.index).ffill()
z=(v-v.rolling(120,min_periods=60).mean())/v.rolling(120,min_periods=60).std()
# Smooth regime-conditioned residual trend: longer 40d horizon and gradual VIX multiplier.
trend=res.rolling(40,min_periods=25).sum()
mult=(1-0.75*np.clip(z,0,2)).clip(-0.5,1.0)
f=trend.mul(mult,axis=0).rank(axis=1,pct=True).sub(.5,axis=0).shift(1)
rows=[]
for dt in f.index:
 a=f.loc[dt]; b=(lp.shift(-10)-lp).loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8 and a[ok].nunique()>1: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
zout=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=zout.ic.dropna()
print('dates',len(q),'avgN',zout.n.mean(),'coverage',zout.n.mean()/15,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turn',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for n in [120,252,756,1260]:
 x=q.tail(n); print('recent',n,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
for h in [1,5,10,20]:
 rr=[]; yy=lp.shift(-h)-lp
 for dt in f.index:
  a,b=f.loc[dt],yy.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8 and a[ok].nunique()>1: rr.append(a[ok].corr(b[ok]))
 x=pd.Series(rr).dropna(); print('horizon',h,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'obs',len(x))
f.to_csv('scripts/miner_2_20340331_vix_smooth_trend40_signal.csv'); zout.to_csv('scripts/miner_2_20340331_vix_smooth_trend40_ic.csv')
