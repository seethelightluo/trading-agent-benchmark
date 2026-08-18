import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base=Path('../persistent/stock_data')
px={}
for s in U:
 d=pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 px[s]=d
p=pd.DataFrame(px).sort_index().ffill()
# restrict information/forward evaluation to dates whose forward endpoint is <= current date
cut=pd.Timestamp('2028-03-09')
rets=p.pct_change(20)
# Relative momentum: own 20d return minus cross-sectional median, then lagged one day
fac=rets.sub(rets.median(axis=1),axis=0).shift(1)
results={}
for h in [1,5,10,20]:
 fwd=p.shift(-h)/p-1
 vals=[]; turnovers=[]; ninst=[]
 for dt in fac.index:
  if dt>cut: continue
  x=fac.loc[dt]; y=fwd.loc[dt]
  z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ninst.append(len(z))
 # signal turnover based rank changes over successive dates
 ic=np.array(vals); results[h]=(len(ic),np.nanmean(ic),np.nanstd(ic,ddof=1),np.nanmean(ic)/np.nanstd(ic,ddof=1),np.mean(ic>0),np.mean(ninst))
print('factor=lagged 20d return minus cross-sectional median; cutoff',cut.date())
for h,v in results.items(): print('H',h,'dates %.0f IC %.6f std %.6f ICIR %.6f hit %.4f avgN %.2f'%v)
# coverage and turnover of daily cross-sectional scores
valid=fac.notna().sum(axis=1); print('coverage',valid.loc[:cut].mean()/len(U), 'avg valid',valid.loc[:cut].mean())
rank=fac.rank(axis=1,pct=True); turnover=rank.diff().abs().mean(axis=1).loc[:cut].mean(); print('turnover',turnover)
# regime halves
for label,a,b in [('early','2020-01-01','2024-01-01'),('late','2024-01-01','2028-03-09')]:
 q=[]
 for dt in fac.loc[a:b].index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(label,len(q),np.mean(q),np.mean(q)/np.std(q,ddof=1),np.mean(np.array(q)>0))
