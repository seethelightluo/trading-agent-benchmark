import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).ffill(); lp=np.log(P); r=lp.diff()
# Volatility-normalized medium-horizon reversal. Lagged, equal blend of 10d and 20d losses,
# scaled by trailing 30d realized volatility to compare heterogeneous assets.
vol=r.rolling(30,min_periods=20).std()*np.sqrt(10)
f10=-(lp-lp.shift(10))/(vol+1e-8); f20=-(lp-lp.shift(20))/(vol+1e-8)
f=((f10.rank(axis=1,pct=True)+f20.rank(axis=1,pct=True))/2).rolling(2,min_periods=2).mean().shift(1)
fr=lp.shift(-10)-lp
rows=[]
for dt in f.index:
 a,b=f.loc[dt],fr.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic
print('volnorm_blend_reversal10_20','dates',len(z),'avgN',round(z.n.mean(),2),'coverage',round(z.n.mean()/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'turn',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),4))
for n in [120,252,756]:
 x=q.tail(n); print('recent',n,'ICIR',round(x.mean()/x.std(ddof=1),5),'IC',round(x.mean(),5))
f.to_csv('scripts/miner_2_20330415_volnorm_blend_reversal_signal.csv'); z.to_csv('scripts/miner_2_20330415_volnorm_blend_reversal_ic.csv')
# regime blocks
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2032'),('2033','2033')]:
 x=q.loc[a:b]
 if len(x)>2: print('regime',a,b,'n',len(x),'IC',round(x.mean(),5),'ICIR',round(x.mean()/x.std(ddof=1),5))
