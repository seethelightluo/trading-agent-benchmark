import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index().ffill(); lp=np.log(P); r=lp.diff()
# risk-adjusted medium-term trend; all inputs lagged before forward return
mom=lp-lp.shift(20); vol=r.rolling(20,min_periods=15).std()*np.sqrt(252); raw=(mom/vol).replace([np.inf,-np.inf],np.nan)
f=raw.rank(axis=1,pct=True).shift(1)
y=lp.shift(-10)-lp
rows=[]
for dt in f.index:
 a=f.loc[dt].values;b=y.loc[dt].values;ok=np.isfinite(a)&np.isfinite(b)
 if ok.sum()>=8 and np.unique(a[ok]).size>1: rows.append((dt,np.corrcoef(a[ok],b[ok])[0,1],ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic
print('dates',len(q),'avgN',z.n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
for n in [120,252,756,1260]:
 x=q.tail(n); print('recent',n,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2031'),('2032','2034')]:
 x=q.loc[a:b]; print('regime',a,b,'dates',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1))
turn=f.rank(pct=True).diff().abs().mean(axis=1).mean(); print('coverage',f.notna().mean().mean(),'turnover',turn)
f.to_csv('scripts/miner_2_20340929_riskadjusted_momentum20_signal.csv'); z.to_csv('scripts/miner_2_20340929_riskadjusted_momentum20_ic.csv')
