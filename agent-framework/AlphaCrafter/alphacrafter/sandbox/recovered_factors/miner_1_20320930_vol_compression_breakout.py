import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close']
 px[a]=d
p=pd.DataFrame(px).sort_index()
r=p.pct_change()
# volatility compression breakout: medium trend, rewarded when recent vol is compressed vs long vol
# lag one completed day by construction (signal at t, forward starts t+1)
trend=p.pct_change(20)
v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std()
f=(trend/(v20+1e-12))*(v60/(v20+1e-12)).clip(0.25,4)
# cap cross-sectional extremes only after signal; no future use
rows=[]
for h in [1,5,10,20]:
 ic=[]; ns=[]
 fr=p.shift(-h)/p-1
 for dt in p.index:
  x=f.loc[dt]; y=fr.loc[dt]
  z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8:
   ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 s=pd.Series(ic)
 rows.append((h,len(s),np.mean(ns),s.mean(),s.std(ddof=1),s.mean()/s.std(ddof=1),np.mean(s>0)))
# coverage on dates with >=8 and average valid fraction
valid=f.notna().sum(axis=1); cov=(valid/15).mean(); dates=(valid>=8).sum()
# turnover 10-day rank changes
rank=f.rank(axis=1,pct=True); turn=(rank-rank.shift(10)).abs().mean(axis=1).mean()
print('rows h dates meanN IC std ICIR hit')
for x in rows: print('%d %d %.2f %+.6f %.6f %+.6f %.4f'%x)
print('coverage %.4f valid_dates_ge8 %d total_dates %d turnover10 %.4f'%(cov,dates,len(p),turn))
for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2032')]:
 z=[]; fr=p.shift(-10)/p-1
 for dt in p.loc[lo:hi].index:
  q=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 print('regime',lo,hi,len(z),np.mean(z) if z else np.nan, (np.mean(z)/np.std(z,ddof=1)) if len(z)>1 else np.nan)
