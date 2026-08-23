import os, numpy as np, pandas as pd
from scipy.stats import spearmanr

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index()
 px[s]=d
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# Relative strength: asset 20-day trend minus contemporaneous cross-sectional median trend.
# This isolates leadership/lagging relative to the available benchmark universe.
ret20=P/P.shift(20)-1
fac=ret20.sub(ret20.median(axis=1),axis=0)
fwd=P.shift(-10)/P-1
rows=[]
for dt in fac.index:
 x=fac.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  rows.append((dt,ic,len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
# same horizon admission stats
mean=a.ic.mean(); sd=a.ic.std(ddof=1); icir=mean/sd*np.sqrt(252/10)
# signal turnover: average fraction whose sign/rank changes across dates
rank=fac.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna().mean()
coverage=a.n.mean()/15
print('dates',len(a),'mean_n',a.n.mean(),'coverage',coverage)
print('IC',mean,'ICIR',icir,'hit', (a.ic>0).mean(),'turnover',turn)
for name,lo,hi in [('2020-24','2020','2024-12-31'),('2025-26','2025','2026-12-31'),('2027-28','2027','2028-12-31'),('2029YTD','2029','2029-12-31')]:
 q=a.loc[(a.index>=lo)&(a.index<=hi),'ic']; print(name,len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252/10) if len(q)>1 else np.nan)
# decay diagnostic (same factor against horizons)
for h in [5,10,20]:
 fw=P.shift(-h)/P-1; vals=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,len(vals),np.nanmean(vals),np.nanmean(vals)/np.nanstd(vals,ddof=1)*np.sqrt(252/h))
# artifact with date and factor values
out=fac.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna()
out.to_csv('scripts/miner_2_20290809_relative_strength_20d_signal.csv',index=False)
