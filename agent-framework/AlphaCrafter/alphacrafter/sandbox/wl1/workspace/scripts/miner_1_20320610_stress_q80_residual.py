import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 d=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].astype(float).sort_index()
 px[s]=d
p=pd.DataFrame(px).sort_index(); r=np.log(p).diff(); bench=r.mean(axis=1)
# Q80 residual dispersion, stress-or-dispersion, 10d horizon; all signal inputs shifted by one day
res=r.sub(bench,axis=0)
disp=res.std(axis=1)
bench20=bench.rolling(20).sum()
q80=disp.rolling(120,min_periods=80).quantile(.80)
active=(bench20<0)|(disp>q80)
fac=(-res.rolling(10).sum().div(res.rolling(60).std())).where(active).shift(1)
fwd=np.log(p.shift(-10)/p)
ics=[]; dates=[]; nobs=[]
for dt in fac.index:
 x=fac.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  z=spearmanr(x[ok],y[ok]).statistic
  if np.isfinite(z): ics.append(z);dates.append(dt);nobs.append(ok.sum())
ics=np.array(ics)
print('factor=stress_or_q80_residual_pullback_10d')
print('dates',len(ics),'calendar',len(p),'avg_n',np.mean(nobs),'coverage',np.mean(nobs)/15,'IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1),'hit',np.mean(ics>0),'turnover',np.nan)
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032')]:
 z=ics[(np.array(dates)>=pd.Timestamp(a+'-01-01'))&(np.array(dates)<=pd.Timestamp(b+'-12-31'))]
 print(a+'-'+b,len(z), (z.mean() if len(z) else np.nan),(z.mean()/z.std(ddof=1) if len(z)>1 else np.nan))
for h in [1,5,20]:
 fw=np.log(p.shift(-h)/p); zz=[]
 for dt in fac.index:
  ok=fac.loc[dt].notna()&fw.loc[dt].notna()
  if ok.sum()>=8: zz.append(spearmanr(fac.loc[dt][ok],fw.loc[dt][ok]).statistic)
 print('horizon',h,'IC',np.nanmean(zz),'n',len(zz))
out=pd.DataFrame(fac,index=p.index);out.to_csv('scripts/miner_1_20320610_stress_q80_residual_signal.csv')
