import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index().ffill(); lp=np.log(P); r=lp.diff()
# Short-horizon reversal, standardized by recent realized risk and activated after an unusually large move.
move=lp-lp.shift(3); vol=r.rolling(20).std(); shock=(move.abs()/vol.replace(0,np.nan)).clip(0,8)
f=(-move/vol.replace(0,np.nan)).mul((1+0.25*shock),axis=0).shift(1)
y=lp.shift(-10)-lp
rows=[]
for dt in f.index:
 a=f.loc[dt].values;b=y.loc[dt].values;ok=np.isfinite(a)&np.isfinite(b)
 if ok.sum()>=8 and np.unique(a[ok]).size>1: rows.append((dt,np.corrcoef(a[ok],b[ok])[0,1],ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']);q=z.ic
print('dates',len(q),'avgN',z.n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
for n in [120,252,756,1260]:
 x=q.tail(n);print('recent',n,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
f.to_csv('scripts/miner_1_20340929_shock_reversal_signal.csv');z.to_csv('scripts/miner_1_20340929_shock_reversal_ic.csv')
