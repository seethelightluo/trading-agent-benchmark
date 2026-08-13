import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.astype(float) for s in U}).ffill(); r=P.pct_change(); vol=r.rolling(20,min_periods=10).std(); rel=np.log(P/P.shift(5)); raw=-(rel-rel.median(axis=1).values[:,None])/(vol*np.sqrt(5))
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill(); stress=v.rolling(252,min_periods=60).rank(pct=True)
# Short-horizon relative reversal, volatility-normalized and strengthened in high-VIX regimes.
f=(raw*(0.7+0.9*stress.fillna(0.5).values[:,None])).rolling(2,min_periods=2).mean().shift(1).replace([np.inf,-np.inf],np.nan)
fr=np.log(P.shift(-10)/P); rows=[]
for dt in f.index:
 a,b=f.loc[dt],fr.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic
print('dates',len(z),'avgN',z.n.mean(),'assets',len(U),'coverage',z.n.mean()/15,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for n in [120,252,756]: x=q.tail(n); print('recent',n,len(x),x.mean(),x.mean()/x.std(ddof=1))
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2032')]:
 x=q.loc[a:b]; print(a,b,len(x),x.mean(),x.mean()/x.std(ddof=1))
f.to_csv('scripts/miner_1_20321209_vix_stress_reversal5_signal.csv'); z.to_csv('scripts/miner_1_20321209_vix_stress_reversal5_ic.csv'); print('signal_path scripts/miner_1_20321209_vix_stress_reversal5_signal.csv')
