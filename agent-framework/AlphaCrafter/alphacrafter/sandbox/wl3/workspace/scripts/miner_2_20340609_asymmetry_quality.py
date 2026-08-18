import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2034-06-09')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index().loc[:END].ffill(); L=np.log(P); R=L.diff()
up=R.clip(lower=0).rolling(30,min_periods=15).mean(); dn=(-R.clip(upper=0)).rolling(30,min_periods=15).mean(); act=R.abs().rolling(30,min_periods=15).mean()
F=((up-dn)/(act+1e-6)+.25*R.rolling(10,min_periods=8).sum()/(R.rolling(40,min_periods=20).std()+1e-6)).shift(1)
rows=[]; fut={h:L.shift(-h)-L for h in [1,3,5,10,20]}
for dt in F.index:
 for h,y in fut.items():
  a=F.loc[dt]; b=y.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8 and a[ok].nunique()>1: rows.append((dt,h,a[ok].corr(b[ok],method='spearman'),ok.sum()))
z=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,3,5,10,20]:
 q=z[z.h==h].ic.dropna(); print('horizon',h,'dates',len(q),'avgN',z[z.h==h].n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
q=z[z.h==10].set_index('date').ic.dropna()
for n in [120,252,756,1260]:
 x=q.tail(n); print('recent',n,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
print('turn',F.rank(pct=True).diff().abs().mean(axis=1).mean(),'coverage',F.notna().mean().mean(),'dates',len(q),'avg instruments',z[z.h==10].n.mean())
F.to_csv('scripts/miner_2_20340609_asymmetry_quality_signal.csv'); z.to_csv('scripts/miner_2_20340609_asymmetry_quality_ic.csv')
