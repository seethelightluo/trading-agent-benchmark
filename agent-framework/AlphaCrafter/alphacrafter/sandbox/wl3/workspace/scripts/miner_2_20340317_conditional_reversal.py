import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index().ffill(); lp=np.log(P)
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index()['close'].astype(float).reindex(P.index).ffill()
# Conditional relative reversal: reverse the lagged 5d cross-section only when VIX is
# above its lagged 60d median and rising over 3d; otherwise use zero (defensive sleeve).
r5=lp.diff(5); elevated=v>v.rolling(60,min_periods=40).median(); rising=v.diff(3)>0
state=(elevated&rising).astype(float)
f=(-r5*state.values[:,None]).rank(axis=1,pct=True).sub(.5,axis=0).shift(1)
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
print('turn',f.rank(pct=True).diff().abs().mean(axis=1).mean(),'coverage',f.notna().mean().mean(),'dates',len(q),'active',state.mean())
f.to_csv('scripts/miner_2_20340317_conditional_reversal_signal.csv'); z.to_csv('scripts/miner_2_20340317_conditional_reversal_ic.csv')
