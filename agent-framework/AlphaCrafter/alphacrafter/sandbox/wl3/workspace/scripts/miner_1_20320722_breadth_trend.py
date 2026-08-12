import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for s in U}
p=np.log(pd.DataFrame(D).sort_index().ffill()); r=p.diff()
# Breadth-conditioned trend: medium momentum is scaled by contemporaneous
# cross-asset breadth, rewarding confirmed trends and suppressing isolated moves.
mom=p-p.shift(20)
breadth=(r.rolling(20,min_periods=15).sum()>0).mean(axis=1)
# use only trailing breadth information, then smooth and lag one completed session
f=mom.mul((0.5+breadth),axis=0).rolling(5,min_periods=5).mean().shift(1)
fr=p.shift(-10)-p; rows=[]
for dt in f.index:
 a,b=f.loc[dt],fr.loc[dt];ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');q=z.ic
print('dates',len(z),'avgN',z.n.mean(),'assets',len(D),'coverage',z.n.mean()/len(D),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for n in [120,252,756]:
 x=q.tail(n); print('recent',n,len(x),x.mean(),x.mean()/x.std(ddof=1))
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2032')]:
 x=q.loc[a:b]; print(a,b,len(x),x.mean(),x.mean()/x.std(ddof=1))
f.to_csv('scripts/miner_1_20320722_breadth_trend_signal.csv');z.to_csv('scripts/miner_1_20320722_breadth_trend_ic.csv')
