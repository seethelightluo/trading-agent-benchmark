import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-02-10')
p={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index();p[s]=d.close.loc[:end]
c=pd.DataFrame(p); r=c.pct_change(); down=r.where(r<0,0)
# positive trend relative to downside risk; cross-sectional factor
f=c.pct_change(20)/(down.pow(2).rolling(40).mean().pow(.5)*np.sqrt(20)+1e-9)
y=c.shift(-10)/c-1
ics=[]; cov=[]; tr=[]; prev=None
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  a=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(a): ics.append(a);cov.append(len(z)/15)
  q=f.loc[dt].rank(pct=True)
  if prev is not None: tr.append((q-prev).abs().mean())
  prev=q
A=np.array(ics);print('factor=downside_adjusted_momentum_20d dates=%d instruments=15 coverage=%.3f'%(len(A),np.mean(cov)))
print('IC %.5f ICIR %.5f hit %.3f turnover %.5f'%(A.mean(),A.mean()/A.std(ddof=1),np.mean(A>0),np.mean(tr)))
for h in [1,5,10,20]:
 yy=c.shift(-h)/c-1; aa=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,'IC',np.nanmean(aa),'n',len(aa))
