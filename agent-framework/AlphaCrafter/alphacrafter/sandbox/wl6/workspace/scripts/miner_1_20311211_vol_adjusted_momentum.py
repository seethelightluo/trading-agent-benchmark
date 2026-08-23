import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
fs={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 fs[s]=d.close.astype(float)
p=pd.concat(fs,axis=1).sort_index().loc[:'2031-12-10']; r=p.pct_change()
# medium-term momentum scaled by realized risk, with a mild recent-volatility penalty
sig=(p/p.shift(20)-1)/r.rolling(40).std().replace(0,np.nan)
print('candidate=vol_adjusted_momentum_20d; dates=%d instruments=%d last=%s'%(len(p),len(U),p.index.max().date()),flush=True)
for h in [5,10,20]:
 f=p.shift(-h)/p-1; z=[];ns=[];dates=[]
 for dt in sig.index:
  a=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   q=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if pd.notna(q): z.append(q);ns.append(len(a));dates.append(dt)
 z=pd.Series(z,index=dates); print('h=%d dates=%d avg_n=%.2f IC=%.8f ICIR=%.6f hit=%.4f'%(h,len(z),np.mean(ns),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(len(z)),(z>0).mean()),flush=True)
 # recent yearly regime check
 for yr,g in z.groupby(z.index.year):
  if len(g)>30: print(' year=%d n=%d ic=%.6f icir=%.3f'%(yr,len(g),g.mean(),g.mean()/g.std(ddof=1)*np.sqrt(len(g))),flush=True)
print('coverage=%.6f turnover=%.6f'%(sig.notna().sum(axis=1).mean()/15,sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()),flush=True)
