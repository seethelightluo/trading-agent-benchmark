import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 try:D[s]=get_index_daily_data(s,2500)
 except Exception:
  try:D[s]=get_stock_daily_data(s,2500)
  except Exception:D[s]=None
D={s:d for s,d in D.items() if d is not None and len(d)>100}; px=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items()}); r0=px.pct_change(); v=r0.rolling(10).std()
# short-horizon volatility-normalized reversal, all inputs lagged one day
f=-(px.shift(1)/px.shift(3)-1)/v.shift(1); out=[]
for i in range(len(px)-10):
 z=pd.concat([f.iloc[i],px.iloc[i+10]/px.iloc[i]-1],axis=1).dropna()
 if len(z)>=8:out.append((px.index[i],z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
r=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); print('assets',len(D),'dates',len(r),'avg_n',r.n.mean(),'coverage',r.n.mean()/15);print('IC %.6f ICIR %.6f hit %.4f'%(r.ic.mean(),r.ic.mean()/r.ic.std(),(r.ic>0).mean()));
for n,b in [('early','2024-01-01'),('middle','2027-01-01'),('late','2029-01-01')]:
 q=r.loc[b:];print(n,len(q),q.ic.mean(),q.ic.mean()/q.ic.std())
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean());r.to_csv('scripts/miner_2_20291203_volscaled_reversal3_signal.csv')
