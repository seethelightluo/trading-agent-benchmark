import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).ffill(); r=np.log(P).diff()
# Volatility-compressed momentum: intermediate 20d trend, scaled by short/long volatility ratio.
# Low short-vs-long volatility confirms persistent trends; lag one day to avoid lookahead.
mom=np.log(P.shift(3)/P.shift(23))
short=r.rolling(5,min_periods=4).std(); long=r.rolling(40,min_periods=25).std()
f=(mom/(long*np.sqrt(20))*(1-short/(long+1e-12))).shift(1).replace([np.inf,-np.inf],np.nan)
fr=np.log(P.shift(-10)/P); rows=[]
for dt in f.index:
 a,b=f.loc[dt],fr.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
zx=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=zx.ic
print('dates',len(zx),'avgN',zx.n.mean(),'assets',len(U),'coverage',zx.n.mean()/15,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for n in [120,252,756]:
 x=q.tail(n); print('recent',n,len(x),x.mean(),x.mean()/x.std(ddof=1))
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2032'),('2033','2033')]:
 x=q.loc[a:b]; print(a,b,len(x),x.mean(),x.mean()/x.std(ddof=1) if len(x)>1 else np.nan)
f.to_csv('scripts/miner_3_20330106_volcompressed_momentum_signal.csv'); zx.to_csv('scripts/miner_3_20330106_volcompressed_momentum_ic.csv')
print('signal_path scripts/miner_3_20330106_volcompressed_momentum_signal.csv'); print('ic_path scripts/miner_3_20330106_volcompressed_momentum_ic.csv'); print('library_corr null')
