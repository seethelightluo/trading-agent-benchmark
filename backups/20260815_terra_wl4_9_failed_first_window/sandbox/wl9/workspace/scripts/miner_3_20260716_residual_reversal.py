import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'; x=pd.read_csv(f); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close.sort_index()
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
# residualized medium-term reversal: asset 20d return relative to cross-sectional median, inverted
x=r.rolling(20,min_periods=15).sum(); med=x.median(axis=1); fac=-(x.sub(med,axis=0))
# forward close-to-close return, strictly next day
fwd=r.shift(-1)
ics=[]; turns=[]; counts=[]
prev=None
for dt in fac.index:
 a=fac.loc[dt]; b=fwd.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8:
  ics.append(spearmanr(a[ok],b[ok]).statistic); counts.append(ok.sum())
  ranks=a.rank();
  if prev is not None: turns.append((ranks-prev).abs().sum()/(ok.sum()**2))
  prev=ranks
z=np.array(ics); print('N_dates',len(z),'mean_names',np.mean(counts),'coverage',len(z)/len(fac),'IC',np.nanmean(z),'ICIR',np.nanmean(z)/np.nanstd(z,ddof=1),'hit',np.mean(z>0),'turn',np.nanmean(turns))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-15')]:
 q=z # redo date keyed
 vals=[]
 for dt in fac.index:
  if str(dt)[:4]>=lo and str(dt)[:10]<=hi:
   a=fac.loc[dt];b=fwd.loc[dt];ok=a.notna()&b.notna()
   if ok.sum()>=8: vals.append(spearmanr(a[ok],b[ok]).statistic)
 print(lo, np.mean(vals) if vals else None, len(vals))
for h in [1,5,10,20]:
 ff=p.shift(-h).div(p)-1; vals=[]
 for dt in fac.index:
  a=fac.loc[dt];b=ff.loc[dt];ok=a.notna()&b.notna()
  if ok.sum()>=8: vals.append(spearmanr(a[ok],b[ok]).statistic)
 print('decay',h,np.mean(vals),np.mean(vals)/np.std(vals,ddof=1),len(vals))
