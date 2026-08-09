import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={s:pd.read_csv('../persistent/stock_data/'+s+'.csv').set_index('date')['close'] for s in syms}
p=pd.DataFrame(p).sort_index(); r=p.pct_change()
vix=pd.read_csv('../persistent/index_data/VIX.csv').set_index('date')['close'].reindex(p.index).ffill()
# Candidate: VIX-up stress resilience, residualized against unconditional market beta.
# beta is rolling 60d covariance with equal-weight asset return / market variance;
# stress beta uses only VIX-up days, signal is negative stress beta residual (higher=resilience).
mkt=r.mean(axis=1)
up=vix.pct_change()>0
beta=(r.rolling(60).cov(mkt)/mkt.rolling(60).var()).replace([np.inf,-np.inf],np.nan)
stress_cov=r.where(up, np.nan).rolling(60,min_periods=12).cov(mkt.where(up,np.nan))
stress_var=mkt.where(up,np.nan).rolling(60,min_periods=12).var()
sbeta=(stress_cov.div(stress_var,axis=0)).replace([np.inf,-np.inf],np.nan)
# residual stress sensitivity after removing ordinary beta, cross-sectional demean
f=-(sbeta-0.5*beta)
f=f.sub(f.mean(axis=1),axis=0)
dates=p.index
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1; vals=[]; ns=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i],fr.iloc[i]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.asarray(vals); print('H',h,'dates',len(a),'meanIC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'meanN',np.mean(ns))
# regime / recent
fr=p.shift(-1)/p-1; out=[]
for i in range(len(p)-1):
 z=pd.concat([f.iloc[i],fr.iloc[i]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:out.append((dates[i],spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
v=pd.DataFrame(out,columns=['date','ic']).set_index('date')
for a,b in [('2020','2023'),('2024','2027'),('2028','2031'),('2030-08','2031-02')]:
 q=v.loc[a:b,'ic']; print('REG',a,b,len(q),q.mean(),q.mean()/q.std(ddof=1))
rank=f.rank(axis=1,pct=True); print('source_dates',len(p),'instruments',len(syms),'coverage',f.notna().mean().mean(),'turnover',rank.diff().abs().mean().mean(),'persistence',rank.corrwith(rank.shift(1),axis=1).mean())
print('active_dates',f.notna().any(axis=1).sum())
