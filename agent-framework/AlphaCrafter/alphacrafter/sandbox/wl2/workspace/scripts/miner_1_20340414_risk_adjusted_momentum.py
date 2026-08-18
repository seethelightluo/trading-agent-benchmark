import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; cutoff=pd.Timestamp('2034-04-13')
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).sort_values('date').set_index('date'); px[s]=d['close'].replace(0,np.nan)
p=pd.DataFrame(px).sort_index().loc[:cutoff]; r=p.pct_change()
# lagged 20d risk-adjusted momentum; no current-day information
sig=(p.pct_change(20)/(r.rolling(20).std()*np.sqrt(20))).shift(1)
print('through',p.index.max(),'dates',len(p),'assets',len(U))
for h in [10,20,40]:
 f=p.shift(-h)/p-1; vals=[]; ns=[]; ds=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); ds.append(dt)
 x=pd.Series(vals,index=ds).dropna(); print('H',h,'n',len(x),'avgN %.2f'%np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(x.mean(),x.mean()/x.std(),(x>0).mean()))
rank=sig.rank(axis=1,pct=True); turn=(rank.diff().abs().sum(axis=1)/(2*rank.notna().sum(axis=1))).dropna(); print('coverage %.4f turnover %.4f'%(sig.notna().sum(axis=1).mean()/15,turn.mean()))
f=p.shift(-20)/p-1
for label,lo,hi in [('2020-2025','2020','2025-12-31'),('2026-2029','2026','2029-12-31'),('2030-2034','2030','2034-04-13')]:
 x=[]
 for dt in sig.index:
  if not(pd.Timestamp(lo)<=dt<=pd.Timestamp(hi)):continue
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=pd.Series(x).dropna();print(label,'n',len(x),'IC %.6f ICIR %.6f'%(x.mean(),x.mean()/x.std()))
