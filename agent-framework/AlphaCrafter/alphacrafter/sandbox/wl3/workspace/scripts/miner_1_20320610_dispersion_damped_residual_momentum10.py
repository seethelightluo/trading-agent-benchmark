import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for s in U}
p=np.log(pd.DataFrame(D).sort_index().ffill()); r=p.diff()
# Candidate: 10-day momentum residualized to the equal-weight world move,
# damped when cross-sectional dispersion is high (continuation signal),
# and lagged one completed session.
world=r.mean(axis=1); beta=r.rolling(60,min_periods=40).cov(world).div(world.rolling(60,min_periods=40).var(),axis=0)
res=r-beta.mul(world,axis=0)
raw=res.rolling(10,min_periods=10).sum()
disp=r.sub(world,axis=0).std(axis=1).rolling(20,min_periods=15).mean()
dp=disp.rolling(252,min_periods=100).rank(pct=True)
# momentum is strongest when dispersion is moderate/low; cross-sectional rank preserves comparability
f=raw.mul((1.25-dp).clip(0.25,1.25),axis=0).rolling(3,min_periods=3).mean().shift(1)
fr=p.shift(-10)-p
rows=[]
for dt in f.index:
 a,b=f.loc[dt],fr.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic
print('dates',len(z),'avgN',round(z.n.mean(),2),'assets',len(D),'coverage',round(z.n.mean()/len(D),4),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for n in [120,252,756]:
 x=q.tail(n); print('recent',n,len(x),x.mean(),x.mean()/x.std(ddof=1))
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2032')]:
 x=q.loc[a:b]; print('regime',a,b,len(x),x.mean(),x.mean()/x.std(ddof=1))
f.to_csv('scripts/miner_1_20320610_dispersion_damped_residual_momentum10_signal.csv');z.to_csv('scripts/miner_1_20320610_dispersion_damped_residual_momentum10_ic.csv')
