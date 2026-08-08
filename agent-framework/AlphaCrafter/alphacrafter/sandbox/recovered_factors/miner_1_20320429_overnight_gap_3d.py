import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end='2032-04-28'
D={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()
 d=d.loc[:end]
 # overnight gap, then 3-day smoothed gap; sign reversal
 gap=d.open/d.close.shift(1)-1
 # factor positive after negative gap, with normalization by 20d vol
 ret=d.close.pct_change()
 vol=ret.rolling(20).std()
 D[a]=pd.DataFrame({'f':-(gap.rolling(3).mean()/vol), 'r':ret})
dates=sorted(set.intersection(*[set(x.index) for x in D.values()]))
for h in [1,3,5,10,20]:
 vals=[]; turnover=[]; cells=0; total=0
 for i,t in enumerate(dates):
  if i+h>=len(dates): continue
  xs=[]; ys=[]
  for a in assets:
   f=D[a].at[t,'f']; y=D[a].reindex(dates).iloc[i+1:i+h+1]['r'].add(1).prod()-1
   if np.isfinite(f) and np.isfinite(y): xs.append(f);ys.append(y)
  if len(xs)>=8:
   vals.append(spearmanr(xs,ys).statistic); cells+=len(xs)
  total+=len(assets)
  if h==1:
   # cross-sectional rank signal turnover every 10 trading days
   pass
 ic=np.nanmean(vals); sd=np.nanstd(vals,ddof=1); icir=ic/sd*np.sqrt(252) if sd else np.nan
 print(f'H{h} dates={len(vals)} meanN={cells/len(vals):.2f} IC={ic:.6f} ICIR={icir:.6f} hit={np.mean(np.array(vals)>0):.4f}')
# regimes h1 and recent
for label,lo,hi in [('2020-23','2020-01-01','2023-12-31'),('2024-27','2024-01-01','2027-12-31'),('2028-30','2028-01-01','2030-12-31'),('2031+','2031-01-01',end),('recent120',str(pd.Timestamp(end)-pd.Timedelta(days=180)),end)]:
 vals=[]
 for i,t in enumerate(dates[:-1]):
  if not(lo<=str(t.date())<=hi): continue
  xs=[];ys=[]
  for a in assets:
   f=D[a].at[t,'f']; y=D[a].at[dates[i+1],'r']
   if np.isfinite(f) and np.isfinite(y):xs.append(f);ys.append(y)
  if len(xs)>=8: vals.append(spearmanr(xs,ys).statistic)
 ic=np.nanmean(vals); sd=np.nanstd(vals,ddof=1)
 print('REG',label,'n',len(vals),'IC',round(ic,6),'ICIR',round(ic/sd*np.sqrt(252),6) if sd else None)
# turnover rank changes across 10 day decisions
rank=[]
for i,t in enumerate(dates):
 x=np.array([D[a].at[t,'f'] for a in assets]);
 if np.isfinite(x).sum()==15: rank.append(pd.Series(x,index=assets).rank(pct=True).values)
turn=np.mean(np.abs(np.diff(rank,axis=0)[::10])) if len(rank)>10 else np.nan
print('coverage cells',cells,'of',total,'turnover10',turn)
# library max correlation on aligned daily raw signals (factor vs expressions approximated by loading factor json? use factor IDs unavailable; report audit computed vs persisted factor signal impossible generically)
print('VALIDATION_END',end)
