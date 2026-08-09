import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).sort_values('date')
 d=d.query("'2020-01-01'<=date<='2026-07-15'").copy(); gap=d.open/d.close.shift(1)-1
 # continuous shock reversal, winsorized to avoid a few crisis gaps dominating
 f=-gap.clip(-.10,.10)
 # only mildly nonlinear compression, retains ranking except tails
 f2=-np.sign(gap)*np.sqrt(gap.abs().clip(upper=.10))
 # overnight gap relative to recent typical gap, volatility-normalized
 scale=gap.abs().rolling(20,min_periods=10).median(); f3=-(gap/scale).clip(-5,5)
 d['f1']=f;d['f2']=f2;d['f3']=f3;d['r']=d.close.shift(-1)/d.close-1;d['s']=s;rows.append(d[['date','s','f1','f2','f3','r']])
x=pd.concat(rows)
for col in ['f1','f2','f3']:
 vals=[]; ns=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=[col,'r'])
  if len(g)>=8: vals.append(spearmanr(g[col],g.r).statistic);ns.append(len(g))
 a=np.array(vals); print(col,'dates',len(a),'meanN',np.mean(ns),'coverage',len(a)/x.date.nunique(),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0))
 for h in [5,10]:
  vs=[]
  for s,g in x.groupby('s'):
   pass
 # regimes
 for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-15')]:
  z=[]
  for dt,g in x.groupby('date'):
   if str(dt)[:10]>=lo and str(dt)[:10]<=hi:
    g=g.dropna(subset=[col,'r'])
    if len(g)>=8:z.append(spearmanr(g[col],g.r).statistic)
  print(' ',lo,len(z),np.nanmean(z))
 # rank turnover
 z=x.dropna(subset=[col]).copy();z['rank']=z.groupby('date')[col].rank(pct=True);z=z.sort_values(['s','date']);print(' turnover',z.groupby('s').rank.diff().abs().mean().mean() if False else z.groupby('s')['rank'].diff().abs().mean())
