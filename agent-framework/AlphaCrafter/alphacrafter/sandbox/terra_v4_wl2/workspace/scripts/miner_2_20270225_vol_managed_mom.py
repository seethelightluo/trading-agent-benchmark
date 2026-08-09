import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
xs={}
for s in U:
 d=get_stock_daily_data(s,days=2200)
 if d is not None and len(d): xs[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(xs).sort_index(); r=p.pct_change()
# volatility-managed momentum: 60d return divided by 20d realized volatility
sig=pd.DataFrame(index=p.index,columns=p.columns,dtype=float)
for s in p:
 sig[s]=r[s].rolling(60,min_periods=60).sum()/(r[s].rolling(20,min_periods=20).std()*np.sqrt(20))
rows=[]
for i in range(len(p)-1):
 z=pd.concat([sig.iloc[i],r.iloc[i+1]],axis=1).dropna()
 if len(z)>=8: rows.append((p.index[i],z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(x),'avg_n',x.n.mean(),'coverage',x.n.mean()/len(U))
print('IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(ddof=1),'hit',(x.ic>0).mean())
for q in range(4):
 sub=x.iloc[q*len(x)//4:(q+1)*len(x)//4];print('regime',q+1,len(sub),sub.ic.mean(),sub.ic.mean()/sub.ic.std(ddof=1))
print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
out=sig.copy();out.index.name='date';out.to_csv('../persistent/factor_signals_miner_2_20270225_vol_managed_mom.csv')
