import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; px={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv'); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close.sort_index()
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# Defensive downside-volatility factor: inverse trailing 20d RMS of negative returns,
# measured through t and predicting t+1. Lower downside risk receives higher score.
F=pd.DataFrame(index=P.index)
for s in U:
 neg=R[s].where(R[s]<0,0.0)
 F[s]=-np.sqrt((neg**2).rolling(20,min_periods=15).mean())

def calc(h):
 vals=[]
 for dt in P.index:
  if dt not in F.index: continue
  f=F.loc[dt]; y=P.shift(-h).loc[dt]/P.loc[dt]-1
  z=pd.concat([f.rename('f'),y.rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q): vals.append((dt,q,len(z)))
 a=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
 return a
for h in [1,5,10]:
 a=calc(h); print('h',h,'dates',len(a),'mean_n',round(a.n.mean(),2),'IC %.5f ICIR %.5f hit %.3f'%(a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1),(a.ic>0).mean()))
a=calc(1); print('coverage',round(a.n.sum()/(len(a)*15),4))
for yr in range(2020,2027):
 q=a[a.index.year==yr]
 if len(q): print('regime',yr,len(q),'IC %.5f ICIR %.5f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)))
# rank turnover: daily Spearman rank change
rr=F.rank(axis=1,pct=True); ts=[]
for i in range(1,len(rr)):
 x=pd.concat([rr.iloc[i-1],rr.iloc[i]],axis=1).dropna()
 if len(x)>=8: ts.append(1-spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic)
print('rank_turnover',np.mean(ts))
