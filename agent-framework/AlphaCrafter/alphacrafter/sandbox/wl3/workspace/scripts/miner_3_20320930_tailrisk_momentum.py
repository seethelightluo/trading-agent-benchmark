import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).ffill(); r=P.pct_change()
r20=np.log(P/P.shift(20)); neg=r.clip(upper=0); pos=r.clip(lower=0)
down=np.sqrt((neg**2).rolling(40,min_periods=20).mean()); up=np.sqrt((pos**2).rolling(40,min_periods=20).mean())
quality=(up/(down+1e-8)).replace([np.inf,-np.inf],np.nan).clip(.25,4)
f=(r20/down.replace(0,np.nan)*quality.pow(.25)).rolling(3,min_periods=3).mean().shift(1)
fr=np.log(P.shift(-10)/P); rows=[]
for dt in f.index:
 a,b=f.loc[dt],fr.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
zx=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');q=zx.ic
print('dates',len(zx),'avgN',zx.n.mean(),'assets',len(U),'coverage',zx.n.mean()/15,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for n in [120,252,756]:
 x=q.tail(n); print('recent',n,len(x),x.mean(),x.mean()/x.std(ddof=1))
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2032')]:
 x=q.loc[a:b]; print(a,b,len(x),x.mean(),x.mean()/x.std(ddof=1))
f.to_csv('scripts/miner_3_20320930_tailrisk_momentum_signal.csv');zx.to_csv('scripts/miner_3_20320930_tailrisk_momentum_ic.csv')
print('signal_path scripts/miner_3_20320930_tailrisk_momentum_signal.csv');print('ic_path scripts/miner_3_20320930_tailrisk_momentum_ic.csv');print('library_corr null')
