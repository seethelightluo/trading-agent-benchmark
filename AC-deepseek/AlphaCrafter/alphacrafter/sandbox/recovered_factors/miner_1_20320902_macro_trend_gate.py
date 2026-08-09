import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
p=pd.DataFrame(d).sort_index(); v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(p.index).ffill()
trend=p.pct_change(20).shift(1); vg=v.pct_change(5).shift(1)
f=trend.where(vg<0,-trend*.25)
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1; vals=[]; ns=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(vals); print('H',h,'dates',len(a),'meanN',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0)))
rank=f.rank(axis=1,pct=True); print('coverage',f.notna().mean().mean(),'turnover10',np.nanmean((rank-rank.shift(10)).abs().mean(axis=1)))
for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2032')]:
 a=[]
 for dt in p.loc[lo:hi].index:
  z=pd.concat([f.loc[dt],p.shift(-1).loc[dt]/p.loc[dt]-1],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=np.array(a);print(lo,len(a),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1))
