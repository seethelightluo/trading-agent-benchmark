import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:get_stock_daily_data(s, days=5000) for s in U}
close=pd.DataFrame({s:(d.set_index('date')['close'] if d is not None else pd.Series(dtype=float)) for s,d in px.items()}).sort_index(); r=close.pct_change(); asset3=r.rolling(3).sum(); resid=asset3.sub(asset3.median(axis=1),axis=0); vol20=r.rolling(20).std()*np.sqrt(20); factor=(-resid/vol20).shift(1).replace([np.inf,-np.inf],np.nan)
ranks=factor.rank(axis=1,pct=True); to=(ranks-ranks.shift(10)).abs().mean(axis=1).mean(); rows=[]
for h in [1,5,10,20]:
 fr=close.shift(-h)/close-1; vals=[]
 for dt in factor.index:
  x=pd.concat([factor.loc[dt],fr.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(x)>=8: vals.append((dt,x.iloc[:,0].corr(x.iloc[:,1]),len(x)))
 z=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date'); ic=z.ic.mean(); sd=z.ic.std(ddof=1); icir=ic/sd*np.sqrt(252) if sd else np.nan; rows.append((h,len(z),z.n.mean(),ic,icir,(z.ic>0).mean()))
print('cutoff',close.index.max(),'dates',len(close),'assets',close.notna().sum().to_dict()); print('coverage',factor.notna().sum().sum()/factor.size,'turnover10',to)
for x in rows: print('H%d dates=%d avgN=%.2f IC=%+.6f ICIR=%+.6f hit=%.4f'%x)
fr=close.shift(-10)/close-1; vals=[]
for dt in factor.index:
 x=pd.concat([factor.loc[dt],fr.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(x)>=8: vals.append((dt,x.iloc[:,0].corr(x.iloc[:,1])))
z=pd.DataFrame(vals,columns=['date','ic']).set_index('date')
for a in np.array_split(z,3): print('third',len(a),a.mean())
out=factor.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20320920_residual_reversal3_signal.csv',index=False)
