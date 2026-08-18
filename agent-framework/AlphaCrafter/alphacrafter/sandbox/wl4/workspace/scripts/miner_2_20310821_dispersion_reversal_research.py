import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2031-08-20')
px={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 d=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].sort_index()
 px[s]=d[d.index<=end]
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# candidate: negative 10d cross-sectional residual return, active only on high dispersion days
r10=P.pct_change(10); resid=r10.sub(r10.median(axis=1),axis=0)
disp=R.std(axis=1)
# lagged regime measure: dispersion compared to trailing 60d median, all inputs t-1
high=(disp.shift(1)>disp.shift(1).rolling(60,min_periods=30).median())
factor=-resid.shift(1).where(high,0.0)
# compare unconditional and low/high components
for name,F in [('gated',factor),('uncond',-resid.shift(1))]:
 print('\n',name)
 for h in [5,10,20]:
  fwd=P.pct_change(h).shift(-h)
  vals=[]; ns=[]; turns=[]
  prev=None
  for dt in F.index:
   a=F.loc[dt]; b=fwd.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
    vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
   q=a.rank(pct=True)
   if prev is not None: turns.append((q-prev).abs().mean())
   prev=q
  x=np.array(vals); print(h,'dates',len(x),'avgN',np.mean(ns),'IC %.6f'%np.nanmean(x),'ICIR %.6f'%(np.nanmean(x)/np.nanstd(x,ddof=1)*np.sqrt(252/h)),'hit %.3f'%np.mean(x>0),'turn %.4f'%np.nanmean(turns))
 # recency h10
 h=10; fwd=P.pct_change(h).shift(-h); vals=[]
 for dt in F.index[-260:]:
  z=pd.concat([F.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('recent260 h10',len(vals),np.mean(vals),np.mean(vals)/np.std(vals,ddof=1)*np.sqrt(252/10))
print('period',P.index.min(),P.index.max(),'rows',len(P),'disp active',high.mean())
