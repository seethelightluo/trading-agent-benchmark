import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).ffill(); r=np.log(P).diff()
down=r.clip(upper=0).rolling(40,min_periods=20).std()*np.sqrt(252)
raw=np.log(P/P.shift(10))/down.replace(0,np.nan)
f=raw.sub(raw.median(axis=1),axis=0).rolling(3,min_periods=3).mean().shift(1)
fr=np.log(P.shift(-10)/P); rows=[]
for dt in f.index:
 a,b=f.loc[dt],fr.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic
print('dates',len(z),'avgN',z.n.mean(),'assets',len(U),'coverage',z.n.mean()/len(U),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for n in [120,252,756]:
 x=q.tail(n); print('recent',n,len(x),x.mean(),x.mean()/x.std(ddof=1))
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2032')]:
 x=q.loc[a:b]; print(a,b,len(x),x.mean(),x.mean()/x.std(ddof=1))
f.to_csv('scripts/miner_2_20320916_downside10_signal.csv');z.to_csv('scripts/miner_2_20320916_downside10_ic.csv')
