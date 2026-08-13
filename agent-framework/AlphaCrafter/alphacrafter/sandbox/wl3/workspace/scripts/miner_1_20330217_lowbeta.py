import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).ffill(); r=np.log(P).diff(); m=r.mean(axis=1)
# Low-beta defensive exposure: estimate rolling beta to equal-weight benchmark, favor lower beta; lagged and smoothed.
cov=r.rolling(60,min_periods=40).cov(m); vm=m.rolling(60,min_periods=40).var()
beta=cov.div(vm,axis=0); raw=-beta
f=raw.sub(raw.median(axis=1),axis=0).rolling(3,min_periods=3).mean().shift(1)
fr=np.log(P.shift(-10)/P); rows=[]
for dt in f.index:
 a,b=f.loc[dt],fr.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic
print('dates',len(z),'avgN',z.n.mean(),'assets',len(U),'coverage',z.n.mean()/15,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for n in [120,252,756]:
 x=q.tail(n); print('recent',n,len(x),x.mean(),x.mean()/x.std(ddof=1))
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2032'),('2033','2033')]:
 x=q.loc[a:b]; print('regime',a,b,len(x),x.mean(),x.mean()/x.std(ddof=1) if len(x)>1 else np.nan)
f.to_csv('scripts/miner_1_20330217_lowbeta_signal.csv'); z.to_csv('scripts/miner_1_20330217_lowbeta_ic.csv')
print('signal_path scripts/miner_1_20330217_lowbeta_signal.csv'); print('ic_path scripts/miner_1_20330217_lowbeta_ic.csv'); print('library_corr unavailable')
