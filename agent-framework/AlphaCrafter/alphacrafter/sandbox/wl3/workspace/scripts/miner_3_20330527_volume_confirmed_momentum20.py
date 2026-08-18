import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).ffill(); V=pd.DataFrame({s:d.volume.astype(float) for s,d in D.items()}).replace(0,np.nan).ffill()
lp=np.log(P); r=lp.diff()
# Volume-confirmed relative momentum: 20d excess return, weighted by robust volume surprise.
mom=lp.diff(20)-lp.diff(20).median(axis=1).values[:,None]
vs=(np.log(V).rolling(20,min_periods=10).mean()-np.log(V).rolling(120,min_periods=60).mean())
vs=vs.clip(-2,2)
f=(mom*(1+0.35*vs)).shift(1); fr=lp.shift(-10)-lp
rows=[]
for dt in f.index:
 a,b=f.loc[dt],fr.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic
print('volume_confirmed_relative_momentum20','dates',len(z),'avgN',round(z.n.mean(),2),'coverage',round(z.n.mean()/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'turn',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),4))
for n in [120,252,756,1260]:
 x=q.tail(n); print('recent',n,'ICIR',round(x.mean()/x.std(ddof=1),5),'IC',round(x.mean(),5),'hit',round((x>0).mean(),4))
f.to_csv('scripts/miner_3_20330527_volume_confirmed_momentum20_signal.csv'); z.to_csv('scripts/miner_3_20330527_volume_confirmed_momentum20_ic.csv')
