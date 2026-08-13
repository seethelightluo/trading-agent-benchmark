import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for s in U}).sort_index().ffill(); r=np.log(P).diff()
# Slow trend confirmation: 60d relative return, gated by agreement with 20d trend, inverse-vol normalized and lagged.
r20=np.log(P/P.shift(20)); r60=np.log(P/P.shift(60)); med20=r20.median(axis=1);med60=r60.median(axis=1)
vol=r.rolling(40,min_periods=25).std()*np.sqrt(40)
base=(r60.sub(med60,axis=0)/vol)
agree=np.sign(r20.sub(med20,axis=0))*np.sign(r60.sub(med60,axis=0))
f=(base*agree.clip(lower=0)).rolling(3,min_periods=3).mean().shift(1)
fr=np.log(P.shift(-10)/P); rows=[]
for dt in f.index:
 a,b=f.loc[dt],fr.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');q=z.ic
print('dates',len(z),'avgN',z.n.mean(),'assets',len(U),'coverage',z.n.mean()/15,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for n in [120,252,756]:
 x=q.tail(n); print('recent',n,len(x),x.mean(),x.mean()/x.std(ddof=1))
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2032')]:
 x=q.loc[a:b]; print(a,b,len(x),x.mean(),x.mean()/x.std(ddof=1))
f.to_csv('scripts/miner_2_20321223_slow_confirmed_trend_signal.csv');z.to_csv('scripts/miner_2_20321223_slow_confirmed_trend_ic.csv')
print('signal_path scripts/miner_2_20321223_slow_confirmed_trend_signal.csv');print('ic_path scripts/miner_2_20321223_slow_confirmed_trend_ic.csv');print('max_abs_library_correlation null')
