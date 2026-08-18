import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index()
 px[s]=d
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# trend persistence: volatility-scaled medium momentum, multiplied by directional consistency
mom=p.shift(1)/p.shift(21)-1
vol=r.rolling(20).std().shift(1)*np.sqrt(252)
cons=(r.rolling(20).apply(lambda x: np.mean(x>0),raw=True).shift(1)-0.5)*2
f=mom/vol*cons
f=f.replace([np.inf,-np.inf],np.nan)
fwd=p.shift(-10)/p-1
ics=[]; ns=[]; turns=[]; prev=None
for dt in f.index:
 x=f.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(ic): ics.append(ic);ns.append(len(z))
  ranks=x.rank(pct=True)
  if prev is not None:
   q=pd.concat([ranks,prev],axis=1).dropna(); turns.append(np.mean(np.abs(q.iloc[:,0]-q.iloc[:,1])))
  prev=ranks
ics=np.array(ics); n=len(ics)
def stat(a): return (np.mean(a), np.mean(a)/np.std(a,ddof=1)*np.sqrt(len(a))) if len(a)>1 else (np.nan,np.nan)
print('dates',n,'avgN',np.mean(ns),'coverage',np.mean([len(f.loc[d].dropna())/15 for d in f.index]))
print('H10 IC ICIR hit',stat(ics),np.mean(ics>0),'turn',np.mean(turns))
for days in [365,730,1095]:
 a=ics[-days:] if len(ics)>=days else ics
 print('recent',days,stat(a))
# decay horizons using same factor dates
for h in [5,20]:
 yy=p.shift(-h)/p-1; aa=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('H',h,stat(np.array(aa)))
