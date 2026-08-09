import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for a in assets:
 d=pd.read_csv(f'{base}/{a}.csv',parse_dates=['date']).set_index('date')['close'].replace(0,np.nan)
 px[a]=d
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# volatility compression/expansion: inverse recent volatility relative to long baseline, smoothed; interpretable defensive quality
rv5=R.rolling(5,min_periods=4).std(); rv40=R.rolling(40,min_periods=30).std()
f=-(rv5/rv40) # favor compressed assets, test continuation / lower risk
# neutralize common cross-sectional level each date not needed for rank IC
rows=[]
for h in [1,5,10,20]:
  ics=[]; dates=[]; turns=[]; nobs=[]
  prev=None
  for i in range(40,len(P)-h):
    x=f.iloc[i]; y=P.iloc[i+h]/P.iloc[i]-1
    z=pd.concat([x,y],axis=1).dropna();
    if len(z)<8: continue
    ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
    if np.isfinite(ic): ics.append(ic);dates.append(P.index[i]);nobs.append(len(z))
    if prev is not None:
      turns.append(np.mean((x.dropna().rank(pct=True)-prev).abs()))
    prev=x.dropna().rank(pct=True)
  a=np.array(ics); print('H',h,'dates',len(a),'meanN',np.mean(nobs),'IC %.6f ICIR %.6f hit %.4f recent120 %.6f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0),np.mean(a[-120:])))
  for lo,hi,name in [('2020','2023','20-23'),('2024','2027','24-27'),('2028','2030','28-30'),('2031','2032','31')]:
   q=a[(np.array(dates)>=lo)&(np.array(dates)<hi)]
   if len(q): print(' ',name,len(q),'%.6f %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
  print(' turnover',np.mean(turns) if turns else np.nan)
print('coverage',f.notna().mean().mean(), 'dates',len(f))
# sample library correlation proxy against admitted factors that expose expression is not automated
print('candidate vol-ratio inverse')
