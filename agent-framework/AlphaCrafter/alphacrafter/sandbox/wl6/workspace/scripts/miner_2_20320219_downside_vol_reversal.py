import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
fs={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float) for s in U}
p=pd.concat(fs,axis=1).sort_index().loc[:'2032-02-18']; r=p.pct_change()
# Invert recent return normalized by downside deviation: oversold assets rank high.
down=r.where(r<0,0.0).rolling(40,min_periods=20).std()
sig=-(p/p.shift(10)-1)/(down+1e-12)
print('dates=%d instruments=%d cutoff=%s'%(len(p),len(U),p.index.max().date()))
for h in [5,10,20]:
 f=p.shift(-h)/p-1; vals=[]; ns=[]; ds=[]
 for i,dt in enumerate(sig.index[:-h]):
  a=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   q=spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q);ns.append(len(a));ds.append(dt)
 z=pd.Series(vals,index=ds); print('h=%d valid=%d avg_n=%.2f cov=%.4f IC=%.8f ICIR=%.6f hit=%.4f'%(h,len(z),np.mean(ns),np.mean(ns)/15,z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(len(z)),(z>0).mean()))
 if h==10: print('regimes',z.groupby(z.index.year).mean().round(5).to_dict())
print('turnover=%.6f'%(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
