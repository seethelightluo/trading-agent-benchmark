import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2031-09-03'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index(); px[s]=d[d.index<=end]
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); r10=P.pct_change(10); resid=r10.sub(r10.median(axis=1),axis=0); disp=R.std(axis=1)
base=disp.shift(1); med=base.rolling(60,min_periods=30).median(); F=-resid.shift(1).where(base < med, 0.0)
for name,X in [('low_disp',F),('uncond',-resid.shift(1))]:
 print('\n'+name)
 for h in [5,10,20]:
  fwd=P.pct_change(h).shift(-h); vals=[]; ns=[]
  for dt in X.index:
   z=pd.concat([X.loc[dt],fwd.loc[dt]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
  a=np.asarray(vals); ic=np.nanmean(a); ir=ic/np.nanstd(a,ddof=1)*np.sqrt(252/h)
  print(f'H{h} dates {len(a)} avgN {np.mean(ns):.2f} IC {ic:.6f} ICIR {ir:.6f} hit {np.mean(a>0):.3f}')
 vals=[]; fwd=P.pct_change(10).shift(-10)
 for dt in X.index[-260:]:
  z=pd.concat([X.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.asarray(vals); print('recent260 H10',len(a),f'{np.mean(a):.6f}',f'{np.mean(a)/np.std(a,ddof=1)*np.sqrt(252/10):.6f}')
 q=X.rank(pct=True); turnover=q.diff().abs().mean().mean(); print('period',P.index.min().date(),P.index.max().date(),'rows',len(P),'active',float((base<med).mean()),'coverage',float(X.notna().mean().mean()),'turnover',float(turnover))
