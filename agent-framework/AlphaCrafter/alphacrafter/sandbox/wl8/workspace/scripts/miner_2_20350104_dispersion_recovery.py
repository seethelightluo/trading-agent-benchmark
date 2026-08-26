import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}
for s in U:
 d=get_stock_daily_data(s,days=6000)
 if d is None or len(d)<300: d=get_index_daily_data(s,days=6000)
 if d is not None: p[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(p).sort_index(); r=np.log(P).diff()
recovery=P/P.rolling(60).min()-1
neg=r.where(r<0)
down=np.sqrt((neg**2).rolling(30,min_periods=5).mean()).replace(0,np.nan)
cs_disp=r.rolling(20,min_periods=10).std().mean(axis=1)
high=cs_disp > cs_disp.rolling(120,min_periods=30).median()
f=(recovery/down).where(~high, -(recovery/down)).shift(1)
Yall={h:P.shift(-h)/P-1 for h in [1,5,10,20]}
rows=[]; sig=[]
for dt in f.index:
 x=f.loc[dt]; y=Yall[10].loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((dt,x[ok].corr(y[ok],method='spearman'),ok.sum()));sig.append((dt,*x.reindex(U).tolist()))
I=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');v=I.ic
print('factor=dispersion_recovery dates',len(I),'avgN',I.n.mean(),'coverage',I.n.mean()/15)
print('IC10',v.mean(),'ICIR',v.mean()/v.std(),'hit',(v>0).mean(),'recent365',v.tail(365).mean()/v.tail(365).std())
for h,Y in Yall.items():
 a=[]
 for dt in f.index:
  x=f.loc[dt];y=Y.loc[dt];ok=x.notna()&y.notna()
  if ok.sum()>=8:a.append(x[ok].corr(y[ok],method='spearman'))
 print('decay',h,np.nanmean(a))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
I.to_csv('scripts/miner_2_20350104_dispersion_recovery_ic.csv');pd.DataFrame(sig,columns=['date']+U).set_index('date').to_csv('scripts/miner_2_20350104_dispersion_recovery_signal.csv')
