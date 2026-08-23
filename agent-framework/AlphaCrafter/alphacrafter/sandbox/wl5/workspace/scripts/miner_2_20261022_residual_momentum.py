import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END='2026-10-21'
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@END').drop_duplicates('date').set_index('date'); return d.close
p=pd.concat({s:load(s) for s in U},axis=1).sort_index(); R=p.pct_change().values; dates=p.index; n=len(dates); fac=np.full((n,15),np.nan)
for i in range(70,n):
 for j in range(15):
  z=R[i-60:i,[j,1,8]]; ok=np.isfinite(z).all(1)
  if ok.sum()<40: continue
  zz=z[ok]; X=np.c_[np.ones(len(zz)),zz[:,1:]]; b=np.linalg.lstsq(X,zz[:,0],rcond=None)[0]
  q=R[i-10:i,[j,1,8]]; ok=np.isfinite(q).all(1)
  if ok.sum()>=8: fac[i,j]=np.sum(q[ok,0]-b[1]*q[ok,1]-b[2]*q[ok,2]-b[0])
def calc(h):
 fw=np.full_like(R,np.nan); fw[:-h]=p.pct_change(h).values[h:]
 vals=[]; ns=[]; ds=[]
 for i in range(n):
  ok=np.isfinite(fac[i])&np.isfinite(fw[i]);
  if ok.sum()>=8 and len(np.unique(fac[i,ok]))>1: vals.append(spearmanr(fac[i,ok],fw[i,ok]).statistic);ns.append(ok.sum());ds.append(dates[i])
 a=np.array(vals);print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  q=a[[lo<=d.year<=hi for d in ds]];print('regime',lo,hi,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
for h in [1,5,10]:calc(h)
rank=pd.DataFrame(fac,index=dates).rank(axis=1,pct=True);print('coverage',round(np.isfinite(fac).mean(),4),'turnover',round(rank.diff().abs().mean(axis=1).mean(),4),'period',dates[0].date(),dates[-1].date())
rows=[]
for j in range(15):
 for i in range(n):
  if np.isfinite(fac[i,j]) and i>=20:
   mom=np.nansum(R[i-20:i,j]); sd=np.nanstd(R[i-20:i,j]); rev=-np.nansum(R[i-5:i,j])
   if np.isfinite(sd) and sd>0:rows.append([fac[i,j],rev,mom/sd])
c=np.array(rows);print('pooled_corr_reversal_mom',round(spearmanr(c[:,0],c[:,1]).statistic,4),round(spearmanr(c[:,0],c[:,2]).statistic,4))
out=pd.DataFrame(fac,index=dates,columns=U).stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20261022_residual_momentum_signal.csv',index=False)
