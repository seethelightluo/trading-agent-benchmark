import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index().ffill(); r=np.log(P).diff()
# downside-risk efficiency: favor low downside deviation, lagged
neg=r.where(r<0); dd=neg.rolling(30,min_periods=20).std(); f=(-dd).rank(axis=1,pct=True).shift(1); y=np.log(P).shift(-10)-np.log(P)
rows=[]
for dt in f.index:
 a=f.loc[dt].values;b=y.loc[dt].values;ok=np.isfinite(a)&np.isfinite(b)
 if ok.sum()>=8 and np.unique(a[ok]).size>1: rows.append((dt,np.corrcoef(a[ok],b[ok])[0,1],ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');q=z.ic
print('dates',len(q),'avgN',z.n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
for n in [120,252,756]:
 x=q.tail(n);print('recent',n,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2031'),('2032','2034')]:
 x=q.loc[a:b];print('regime',a,b,len(x),x.mean(),x.mean()/x.std(ddof=1))
print('coverage',f.notna().mean().mean(),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
f.to_csv('scripts/miner_2_20340929_downside_risk_signal.csv');z.to_csv('scripts/miner_2_20340929_downside_risk_ic.csv')
