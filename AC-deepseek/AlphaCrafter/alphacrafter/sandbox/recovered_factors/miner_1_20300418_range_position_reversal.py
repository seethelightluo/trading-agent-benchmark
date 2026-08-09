import pandas as pd,numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
 q=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')
 D[a]=q
p=pd.DataFrame({a:q.close for a,q in D.items()}).sort_index(); r=p.pct_change()
# Reversal from 60-observation range extremes: high values mean near high, so negate.
hi=p.rolling(60,min_periods=45).max(); lo=p.rolling(60,min_periods=45).min()
f=(-(p-lo)/(hi-lo)).replace([np.inf,-np.inf],np.nan).shift(1)
print('assets',len(assets),'dates',len(p),'coverage',f.notna().stack().mean())
for h in [1,5,10,20]:
 fr=p.pct_change(h).shift(-h); ic=[]; ns=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(ic); print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(np.mean(a),np.mean(a)/np.std(a,ddof=1),np.mean(a>0)))
# regimes and recent
for lab,start,end in [('2020-24','2020','2024-12-31'),('2025-27','2025','2027-12-31'),('2028-29','2028','2029-12-31'),('latest120','2029-11-01','2030-04-17')]:
 fr=p.pct_change(5).shift(-5); aa=[]
 for dt in p.index[(p.index>=start)&(p.index<=end)]:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 aa=np.array(aa); print(lab,len(aa), 'IC %.6f ICIR %.6f'%(np.mean(aa),np.mean(aa)/np.std(aa,ddof=1)))
ranks=f.rank(axis=1,pct=True);print('turn10',np.nanmean((ranks-ranks.shift(10)).abs().mean(axis=1)))
