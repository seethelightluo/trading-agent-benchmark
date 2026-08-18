import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).ffill(); lp=np.log(P); r=lp.diff()
# Short-horizon reversal conditioned on unusually high recent range: fade 3d return only after a 20d volatility shock.
ret3=lp-lp.shift(3); v5=r.rolling(5,min_periods=5).std(); v30=r.rolling(30,min_periods=20).std(); shock=v5/(v30+1e-8)
f=(-ret3/(v30*np.sqrt(3)+1e-8)*(shock>1.15).astype(float)).rolling(2,min_periods=2).mean().shift(1)
fr=lp.shift(-10)-lp
rows=[]
for dt in f.index:
 a,b=f.loc[dt],fr.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic
print('volshock_reversal3','dates',len(z),'avgN',round(z.n.mean(),2),'coverage',round(z.n.mean()/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'turn',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),4))
for n in [120,252,756]:
 x=q.tail(n); print('recent',n,'ICIR',round(x.mean()/x.std(ddof=1),5),'IC',round(x.mean(),5))
f.to_csv('scripts/miner_3_20330401_volshock_reversal_signal.csv'); z.to_csv('scripts/miner_3_20330401_volshock_reversal_ic.csv')
