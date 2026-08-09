import numpy as np, pandas as pd, glob
from pathlib import Path

root=Path('../persistent')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(name, macro=False):
 p=(root/'index_data'/f'{name}.csv') if macro else (root/'stock_data'/f'{name}.csv')
 d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date')
 return d['close'].astype(float)
prices=pd.concat({s:load(s) for s in syms},axis=1)
dxy=load('DXY',True)
ret=prices.pct_change(); dr=dxy.pct_change()
# Factor: 5-session residual return to trailing 60-session DXY beta, fully date aligned.
beta=ret.rolling(60,min_periods=45).cov(dr).div(dr.rolling(60,min_periods=45).var(),axis=0)
r5=prices.pct_change(5); d5=dxy.pct_change(5)
f=r5-beta.mul(d5,axis=0)
f=f.replace([np.inf,-np.inf],np.nan)
fwd=prices.shift(-1).div(prices)-1
rows=[]
for dt in f.index:
 x=f.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8: rows.append((dt,float(z.iloc[:,0].corr(z.iloc[:,1])),len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('factor=5d DXY-beta residual; dates',len(r),'mean names',r.n.mean(),'coverage',r.n.sum()/(len(r)*15))
print('IC %.6f ICIR %.6f hit %.4f turnover %.4f'%(r.ic.mean(),r.ic.mean()/r.ic.std(),(r.ic>0).mean(),np.nan))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 q=r.loc[a:b];print(a,'dates',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(),(q.ic>0).mean()))
for h in [1,5,10]:
 yy=prices.shift(-h).div(prices)-1; rr=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(float(z.iloc[:,0].corr(z.iloc[:,1])))
 q=pd.Series(rr); print('horizon',h,'dates',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std()))
# rank turnover of signal across adjacent dates
ranks=f.rank(axis=1,pct=True); common=ranks.notna().sum(axis=1)>=8
print('rank turnover',ranks.loc[common].diff().abs().mean(axis=1).mean())
print('validation_end',r.index.max().date())
