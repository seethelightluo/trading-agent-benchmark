import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for s in U}).sort_index().ffill()
R=np.log(P).diff()
macro=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.astype(float).reindex(P.index).ffill()
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').close.astype(float).reindex(P.index).ffill()
# Risk-regime conditioned medium momentum: asset 20d momentum, damped in elevated VIX
# and when DXY is strengthening. All conditioning inputs are lagged before forward returns.
mom=P/P.shift(20)-1
vp=macro.rolling(252,min_periods=100).rank(pct=True)
dxytrend=dxy.pct_change(20)
dp=dxytrend.rolling(252,min_periods=100).rank(pct=True)
reg=(1-vp)*(1-dp)+0.15
f=mom.mul(reg,axis=0).rolling(5,min_periods=5).mean().shift(1)
fr=np.log(P.shift(-10)/P); rows=[]
for dt in f.index:
 a,b=f.loc[dt],fr.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic
print('dates',len(z),'avgN',z.n.mean(),'assets',len(U),'coverage',z.n.mean()/len(U),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for n in [120,252,756]:
 x=q.tail(n);print('recent',n,len(x),x.mean(),x.mean()/x.std(ddof=1))
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2032')]:
 x=q.loc[a:b];print(a,b,len(x),x.mean(),x.mean()/x.std(ddof=1))
f.to_csv('scripts/miner_3_20320722_macro_regime_momentum_signal.csv');z.to_csv('scripts/miner_3_20320722_macro_regime_momentum_ic.csv')
