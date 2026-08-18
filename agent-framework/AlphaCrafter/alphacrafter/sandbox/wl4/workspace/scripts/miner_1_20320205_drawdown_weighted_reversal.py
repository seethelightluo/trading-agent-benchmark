import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-02-04')
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 px[s]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index(); r=p.pct_change();
# Candidate: residual reversal, volatility-normalized, with stronger weight for assets in deep trailing drawdown.
r10=p.pct_change(10); resid=r10.sub(r10.median(axis=1),axis=0)
vol=r.rolling(40,min_periods=20).std(); dd=p/p.rolling(60,min_periods=30).max()-1
sig=(-(resid.rolling(3,min_periods=3).mean())/vol) * (1 + 0.75*(-dd).clip(0,0.5))
sig=sig.shift(1)
rows=[]
for h in [5,10,20]:
 fwd=p.shift(-h)/p-1
 ics=[]; ns=[]; turnovers=[]
 for dt in sig.index:
  x=sig.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8:
   ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 # rank turnover, aligned valid dates
 ranks=sig.rank(axis=1,pct=True); turnovers=ranks.diff().abs().mean(axis=1).dropna()
 a=np.array(ics); ic=a.mean(); ir=ic/a.std(ddof=1) if len(a)>1 else np.nan
 print(f'H{h}: dates={len(a)} avgN={np.mean(ns):.2f} IC={ic:.6f} ICIR={ir:.6f} hit={np.mean(a>0):.4f} turnover={turnovers.mean():.4f}')
 for n in [365,730,1095]:
  sub=[]
  dates=sig.index
  start=cut-pd.Timedelta(days=n)
  for dt in dates[(dates>=start)&(dates<=cut)]:
   z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
   if len(z)>=8: sub.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  aa=np.array(sub); print(f' recent{n}d: dates={len(aa)} IC={aa.mean():.6f} ICIR={aa.mean()/aa.std(ddof=1):.6f}')
print('coverage=',sig.notna().mean().mean(),'overall_dates=',sig.dropna(how='all').shape[0],'assets=',len(U))
