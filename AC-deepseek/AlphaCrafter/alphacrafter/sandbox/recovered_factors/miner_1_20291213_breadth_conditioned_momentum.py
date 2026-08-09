import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A}).sort_index(); r=p.pct_change()
# Relative 20d momentum, activated only when broad cross-asset breadth confirms trend; lagged.
m20=p.pct_change(20); breadth=(m20>0).mean(axis=1)
f=(m20 - m20.mean(axis=1).values[:,None])*((breadth-0.5).abs()+0.25)*np.sign(breadth-0.5).replace(0,np.nan).values[:,None]
f=pd.DataFrame(f,index=p.index,columns=p.columns).shift(1)
print('raw dates',len(p),'assets',len(A),'cells',int(f.notna().sum().sum()),'coverage',f.notna().sum().sum()/f.size,'mean_valid',f.notna().sum(axis=1).mean())
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; z=[];ns=[]
 for d in f.index:
  ok=f.loc[d].notna()&y.loc[d].notna()
  if ok.sum()>=8:
   q=spearmanr(f.loc[d,ok],y.loc[d,ok]).statistic
   if np.isfinite(q): z.append(q);ns.append(ok.sum())
 s=pd.Series(z);print('h=%d dates=%d meanN=%.2f IC=%.6f ICIR=%.6f hit=%.4f'%(h,len(s),np.mean(ns),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
print('turnover10',((f.rank(axis=1,pct=True)-f.rank(axis=1,pct=True).shift(10)).abs().mean(axis=1)).mean())
for st,en in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-12')]:
 y=p.shift(-10)/p-1;z=[]
 for d in f.loc[st:en].index:
  ok=f.loc[d].notna()&y.loc[d].notna()
  if ok.sum()>=8:
   q=spearmanr(f.loc[d,ok],y.loc[d,ok]).statistic
   if np.isfinite(q):z.append(q)
 s=pd.Series(z);print('regime',st,en,'dates',len(s),'IC %.6f ICIR %.6f'%(s.mean(),s.mean()/s.std(ddof=1)))
