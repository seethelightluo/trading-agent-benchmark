import pandas as pd, numpy as np
from scipy.stats import spearmanr
AS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2031-09-03'); START=pd.Timestamp('2026-07-16')
D={a:pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index() for a in AS}
idx=sorted(set.intersection(*[set(x.loc[(x.index>=START)&(x.index<=CUT)].index) for x in D.values()]))
c=pd.DataFrame({a:D[a].reindex(idx).close for a in AS}); r=c.pct_change(); med=r.median(1)
beta=r.rolling(60,45).cov(med).div(med.rolling(60,45).var(),axis=0); resid=r-beta.mul(med,axis=0)
# One idea: recent residual downside severity, scaled by its own trailing downside semivolatility.
# Negative sign makes shallow/reduced downside relative to normal rank higher.
down=resid.where(resid<0); sem=np.sqrt(down.pow(2).rolling(60,45).mean())
sig=-down.shift(1).rolling(5,4).mean().div(sem.shift(1)).replace([np.inf,-np.inf],np.nan)
print('IDEA residual_downside_severity_relief_5_60obs; cutoff',CUT.date())
print('dates',len(idx),'signal cells',int(sig.notna().sum().sum()),'/',sig.size,'coverage',sig.notna().mean().mean())
all_ics={}
for q in [1,5,10,20]:
 f=c.shift(-q).div(c).sub(1); ics=[]; ns=[]
 for t in idx:
  z=pd.concat([sig.loc[t],f.loc[t]],axis=1).dropna()
  if len(z)>=8:
   ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 x=np.array(ics); all_ics[q]=(x,ns)
 print('horizon',q,'ic_dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4),'mean_n',round(np.mean(ns),2))
for label,lo,hi in [('2026-2029',pd.Timestamp('2026-07-16'),pd.Timestamp('2029-12-31')),('2030-current',pd.Timestamp('2030-01-01'),CUT)]:
 q=10; f=c.shift(-q).div(c).sub(1); x=[]
 for t in idx:
  if not lo<=t<=hi: continue
  z=pd.concat([sig.loc[t],f.loc[t]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=np.array(x);print('regime',label,'n',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
# rank turnover and concentration
turn=[];iqr=[]
for a,b in zip(idx[:-1],idx[1:]):
 x=sig.loc[a];y=sig.loc[b]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:turn.append(np.mean(np.abs(z.iloc[:,0].rank(pct=True)-z.iloc[:,1].rank(pct=True))))
for t in idx:
 z=sig.loc[t].dropna()
 if len(z)>=8: iqr.append(z.quantile(.75)-z.quantile(.25))
print('turnover',round(np.mean(turn),6),'comparisons',len(turn),'median_iqr',round(np.median(iqr),6))
