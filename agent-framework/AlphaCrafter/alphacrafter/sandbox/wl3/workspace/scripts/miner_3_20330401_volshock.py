import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).ffill(); lp=np.log(P); r=lp.diff()
v5=r.rolling(5,min_periods=5).std(); v60=r.rolling(60,min_periods=40).std()
# Volatility shock mean reversion: favor assets whose short-term shock is modest relative to baseline,
# with a mild 20d momentum confirmation, lagged and smoothed.
shock=np.log((v5+1e-8)/(v60+1e-8)); mom=lp-lp.shift(20)
f=(-shock + 0.25*mom/(v60*np.sqrt(20)+1e-8)).rolling(3,min_periods=3).mean().shift(1)
fr=lp.shift(-10)-lp
rows=[]
for dt in f.index:
 a,b=f.loc[dt],fr.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic
print('volshock_calm_momentum','dates',len(z),'avgN',round(z.n.mean(),2),'coverage',round(z.n.mean()/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'turn',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),4))
for n in [120,252,756]:
 x=q.tail(n); print('recent',n,'ICIR',round(x.mean()/x.std(ddof=1),5),'IC',round(x.mean(),5))
f.to_csv('scripts/miner_3_20330401_volshock_signal.csv'); z.to_csv('scripts/miner_3_20330401_volshock_ic.csv')
