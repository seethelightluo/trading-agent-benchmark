import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for s in U}).sort_index().ffill()
r=np.log(P).diff(); csmed=r.median(axis=1)
# Macro-residual trend: 20d asset return relative to cross-asset median, remove common VIX shock exposure
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].astype(float).reindex(P.index).ffill()
vshock=np.log(v).diff().rolling(5,min_periods=3).sum()
ret20=np.log(P/P.shift(20)); common=ret20.median(axis=1)
# cross-sectional beta to VIX using rolling daily returns; residualize 20d return by beta * VIX 20d change
beta=pd.DataFrame(index=P.index,columns=U,dtype=float)
for s in U: beta[s]=r[s].rolling(60,min_periods=40).cov(vshock)/vshock.rolling(60,min_periods=40).var()
v20=np.log(v/v.shift(20)); residual=ret20.sub(common,axis=0)-beta.mul(v20,axis=0)
vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
f=(residual/vol).rolling(3,min_periods=3).mean().shift(1)
fr=np.log(P.shift(-10)/P); rows=[]
for dt in f.index:
 a,b=f.loc[dt],fr.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic
print('dates',len(z),'avgN',z.n.mean(),'assets',len(U),'coverage',z.n.mean()/15,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for n in [120,252,756]:
 x=q.tail(n); print('recent',n,len(x),x.mean(),x.mean()/x.std(ddof=1))
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2032')]:
 x=q.loc[a:b]; print(a,b,len(x),x.mean(),x.mean()/x.std(ddof=1))
f.to_csv('scripts/miner_2_20321223_macro_residual_trend_signal.csv');z.to_csv('scripts/miner_2_20321223_macro_residual_trend_ic.csv')
print('signal_path scripts/miner_2_20321223_macro_residual_trend_signal.csv');print('ic_path scripts/miner_2_20321223_macro_residual_trend_ic.csv');print('max_abs_library_correlation null')
