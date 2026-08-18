import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).ffill(); lp=np.log(P); r=lp.diff()
# Volatility term-structure defensive factor: favor assets with low recent volatility
# relative to their own medium baseline, with a slow absolute-volatility tie-break.
v10=r.rolling(10,min_periods=8).std(); v60=r.rolling(60,min_periods=40).std()
f=(-np.log((v10+1e-8)/(v60+1e-8))-0.25*np.log(v60+1e-8)).rolling(3,min_periods=3).mean().shift(1)
fr=lp.shift(-10)-lp
rows=[]
for dt in f.index:
 a,b=f.loc[dt],fr.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic
print('vol_term_defensive','dates',len(z),'avgN',round(z.n.mean(),2),'coverage',round(z.n.mean()/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'turn',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),4))
for n in [120,252,756]:
 x=q.tail(n); print('recent',n,'ICIR',round(x.mean()/x.std(ddof=1),5),'IC',round(x.mean(),5))
f.to_csv('scripts/miner_3_20330415_volterm_signal.csv'); z.to_csv('scripts/miner_3_20330415_volterm_ic.csv')
