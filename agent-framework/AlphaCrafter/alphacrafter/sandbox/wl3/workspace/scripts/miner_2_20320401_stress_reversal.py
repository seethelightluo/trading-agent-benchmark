import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={s:get_stock_daily_data(s,days=5000) for s in U}
px=pd.DataFrame({s:(d.set_index('date')['close'] if d is not None else pd.Series(dtype=float)) for s,d in frames.items()}).sort_index().ffill()
# Candidate: stress-conditioned short-term reversal. Stress is breadth below 20d MA; higher score means stronger reversal in stressed tape.
r=np.log(px).diff(); ma=px/px.rolling(20,min_periods=15).mean()-1
breadth=(ma<0).sum(axis=1)/ma.notna().sum(axis=1)
stress=(breadth-breadth.rolling(60,min_periods=30).mean())/breadth.rolling(60,min_periods=30).std()
# lag observable factor: 5d reversal, amplified by current stress; cross-sectional demean
raw=-r.rolling(5,min_periods=5).sum().div(r.rolling(20,min_periods=15).std())
factor=raw.mul((1+stress.clip(-2,2)).clip(0,3),axis=0)
# factor is lagged one day, forward H10
f=factor.shift(1); fw=np.log(px.shift(-10)/px)
rows=[]
for dt in f.index:
 x=f.loc[dt]; y=fw.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(ic),'avgN',ic.n.mean(),'coverage',ic.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(ic.ic.mean(),ic.ic.mean()/ic.ic.std(),(ic.ic>0).mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for n in [60,120,252,756]:
 q=ic.tail(n).ic; print('recent',n,'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std()))
print('period',ic.index.min(),ic.index.max())
# save recoverable signal artifact
factor.loc[ic.index].to_csv('scripts/miner_2_20320401_stress_reversal_signal.csv')
