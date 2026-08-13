import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).ffill(); r=np.log(P).diff()
# Drawdown rebound: favor assets that have recently rebounded from a 60-day local drawdown, while controlling for 20-day trend.
dd=P/P.rolling(60,min_periods=30).max()-1
reb=r.rolling(5,min_periods=5).sum()/(r.rolling(40,min_periods=25).std()*np.sqrt(5)+1e-12)
raw=reb*(-dd.clip(upper=0))
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
f.to_csv('scripts/miner_3_20330203_drawdown_rebound_signal.csv'); z.to_csv('scripts/miner_3_20330203_drawdown_rebound_ic.csv')
print('signal_path scripts/miner_3_20330203_drawdown_rebound_signal.csv'); print('ic_path scripts/miner_3_20330203_drawdown_rebound_ic.csv'); print('library_corr unavailable')
