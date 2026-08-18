import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).ffill(); lp=np.log(P); r=lp.diff()
# Candidate: dispersion-gated short-horizon reversal. Fade 5-day relative return,
# volatility scale it, and activate only when trailing 20-day cross-asset dispersion is above its 120-day median.
rel=lp.diff(5)-lp.diff(5).median(axis=1).values[:,None]
vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
disp=r.rolling(20,min_periods=15).std().median(axis=1)
gate=(disp>disp.rolling(120,min_periods=60).median()).astype(float)
f=(-rel/(vol+1e-8)).mul(gate,axis=0).shift(10).shift(1)
fr=lp.shift(-10)-lp
rows=[]
for dt in f.index:
 a,b=f.loc[dt],fr.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8 and a[ok].nunique()>1: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic
print('dispersion_gated_reversal5','dates',len(z),'avgN',round(z.n.mean(),2),'coverage',round(z.n.mean()/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'turn',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),4))
for n in [120,252,756,1260]:
 x=q.tail(n); print('recent',n,'ICIR',round(x.mean()/x.std(ddof=1),5),'IC',round(x.mean(),5),'hit',round((x>0).mean(),4))
f.to_csv('scripts/miner_2_20330513_dispersion_gated_reversal5_signal.csv'); z.to_csv('scripts/miner_2_20330513_dispersion_gated_reversal5_ic.csv')
