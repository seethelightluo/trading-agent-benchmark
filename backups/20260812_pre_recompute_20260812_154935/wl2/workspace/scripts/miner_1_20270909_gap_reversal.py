import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  x=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index()
  D[s]=x
# gap reversal: prior close to today's open, scaled by prior 20d ATR; signal known at close today
rows=[]
for s,x in D.items():
 prev=x.close.shift(1)
 gap=(x.open-prev)/prev
 atr=(x.high-x.low).rolling(20,min_periods=10).mean()/prev
 fac=(-gap/atr).replace([np.inf,-np.inf],np.nan)
 # forward close-to-close return
 fr=x.close.shift(-1)/x.close-1
 for dt,v in fac.items(): rows.append((dt,s,v,fr.get(dt,np.nan)))
z=pd.DataFrame(rows,columns=['date','sym','factor','fwd']).dropna()
ics=[]; ns=[]; turnovers=[]
for dt,g in z.groupby('date'):
 if len(g)>=8:
  c=spearmanr(g.factor,g.fwd).statistic
  if np.isfinite(c): ics.append(c); ns.append(len(g))
# rank turnover on common dates
r=z.pivot(index='date',columns='sym',values='factor').rank(axis=1,pct=True)
turn=r.diff().abs().mean(axis=1).dropna()
a=np.array(ics); mean=a.mean(); ir=mean/a.std(ddof=1)
print('idea=ATR-scaled overnight gap reversal')
print('dates',len(a),'avgN',np.mean(ns),'coverage',len(z)/(sum(len(x) for x in D.values())),'daily_ic',mean,'daily_icir',ir,'hit',np.mean(a>0),'turnover',turn.mean())
for h in [3,5,10]:
 # recompute fwd h close observations
 zz=[]
 for s,x in D.items():
  prev=x.close.shift(1); gap=(x.open-prev)/prev; atr=(x.high-x.low).rolling(20,min_periods=10).mean()/prev
  fac=-gap/atr; fr=x.close.shift(-h)/x.close-1
  zz.append(pd.DataFrame({'factor':fac,'fwd':fr,'sym':s}))
 q=pd.concat(zz).reset_index().dropna(); aa=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8:
   c=spearmanr(g.factor,g.fwd).statistic
   if np.isfinite(c): aa.append(c)
 aa=np.array(aa); print('h',h,'ic',aa.mean(),'icir',aa.mean()/aa.std(ddof=1),'dates',len(aa))
# recent regime
for lo,hi in [('2020-01-01','2023-12-31'),('2024-01-01','2026-12-31'),('2027-01-01','2027-08-30')]:
 aa=[]
 for dt,g in z[(z.date>=lo)&(z.date<=hi)].groupby('date'):
  if len(g)>=8:
   c=spearmanr(g.factor,g.fwd).statistic
   if np.isfinite(c): aa.append(c)
 aa=np.array(aa); print(lo,hi,len(aa),aa.mean() if len(aa) else np.nan,(aa.mean()/aa.std(ddof=1)) if len(aa)>1 else np.nan)
