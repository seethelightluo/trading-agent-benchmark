import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for s in U}
p=np.log(pd.DataFrame(P).sort_index().ffill()); r=p.diff(); world=r.mean(axis=1)
beta=r.rolling(60,min_periods=40).cov(world).div(world.rolling(60,min_periods=40).var(),axis=0); res=r-beta.mul(world,axis=0)
dxy=np.log(pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').close.astype(float)).reindex(p.index).ffill()
trend=dxy.diff(20); strength=trend.abs().rolling(252,min_periods=100).rank(pct=True)
# Idiosyncratic 20d momentum, attenuated during strong dollar trends; lagged and smoothed.
f=res.rolling(20,min_periods=20).sum().mul((1.25-strength).clip(.25,1.25),axis=0).rolling(5,min_periods=5).mean().shift(1)
fr=p.shift(-10)-p; rows=[]
for dt in f.index:
 a,b=f.loc[dt],fr.loc[dt];ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');q=z.ic
print('dates',len(z),'avgN',round(z.n.mean(),2),'assets',len(U),'coverage',round(z.n.mean()/len(U),4),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for n in [120,252,756]:
 x=q.tail(n);print('recent',n,len(x),x.mean(),x.mean()/x.std(ddof=1))
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2032')]:
 x=q.loc[a:b];print('regime',a,b,len(x),x.mean(),x.mean()/x.std(ddof=1))
f.to_csv('scripts/miner_1_20320610_dxy_residual_momentum20_signal.csv');z.to_csv('scripts/miner_1_20320610_dxy_residual_momentum20_ic.csv')
