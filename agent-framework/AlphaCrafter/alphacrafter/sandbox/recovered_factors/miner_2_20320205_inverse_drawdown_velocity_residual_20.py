import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
# One close-only idea: inverse residual drawdown-velocity resilience.
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={a:pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').close for a in assets}
P=pd.DataFrame(px).sort_index().ffill(); R=P.pct_change(); dates=P.index
# 20d drawdown velocity: latest five-day worsening of the 20d peak drawdown.
# Residualize each date cross-sectionally versus current 20d vol and 20d trend.
f=pd.DataFrame(np.nan,index=dates,columns=assets)
for t in range(25,len(dates)):
  w=P.iloc[t-20:t+1]; dd=w.iloc[-1]/w.max()-1; dd5=w.iloc[-6]/w.iloc[:-5].max()-1
  raw=-(dd-dd5) # positive where drawdown is stabilizing; inverse orientation evaluated below too
  vol=R.iloc[t-19:t+1].std(); trend=P.iloc[t]/P.iloc[t-20]-1
  ok=raw.notna()&vol.notna()&trend.notna()
  if ok.sum()>=8:
   X=np.c_[np.ones(ok.sum()),vol[ok],trend[ok]]
   b=np.linalg.lstsq(X,raw[ok],rcond=None)[0]
   f.loc[dates[t],ok]=raw[ok]-X@b
# test both orientation only; selected sign is determined from full historical validation (report selection caveat)
for sign,label in [(1,'stabilization_residual'),(-1,'inverse_stabilization_residual')]:
 s=f*sign
 print('\nCANDIDATE',label)
 for h in [1,5,10,20]:
  ics=[]; ns=[]
  for t in range(len(dates)-h):
   x=s.iloc[t]; y=P.iloc[t+h]/P.iloc[t]-1; ok=x.notna()&y.notna()
   if ok.sum()>=8:
    z=spearmanr(x[ok],y[ok]).statistic
    if np.isfinite(z): ics.append(z);ns.append(ok.sum())
  a=np.array(ics); ic=a.mean(); ir=ic/a.std(ddof=1) if a.std(ddof=1)>0 else np.nan
  print(f'h={h} IC={ic:.6f} ICIR={ir:.6f} hit={(a>0).mean():.4f} dates={len(a)} avgN={np.mean(ns):.2f}')
# turnover and coverage selected raw orientation
rank=s.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).mean()
print('coverage',s.notna().mean().mean(),'cells',int(s.notna().sum().sum()),'turnover',turn,'endpoint',dates.max().date())
# subperiod 20d for raw positive orientation
for lo,hi in [('2020-01-01','2026-12-31'),('2027-01-01','2032-02-04')]:
  a=[]
  for t in range(len(dates)-20):
   if not (pd.Timestamp(lo)<=dates[t]<=pd.Timestamp(hi)):continue
   x=f.iloc[t];y=P.iloc[t+20]/P.iloc[t]-1;ok=x.notna()&y.notna()
   if ok.sum()>=8:a.append(spearmanr(x[ok],y[ok]).statistic)
  a=np.array(a);print('regime',lo,hi,'n',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
print('NOVELTY: not computed: no canonical historical signal panels are persisted for every admitted library factor; therefore this candidate cannot satisfy binding correlation evidence and will not be persisted regardless of IC.')
