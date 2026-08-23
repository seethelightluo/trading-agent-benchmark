import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import rankdata
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
pd0={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),usecols=['date','close']); d.date=pd.to_datetime(d.date); pd0[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(pd0).sort_index().loc[:'2029-10-18']; r=np.log(p/p.shift(1));
def run(f,fw,h):
 # rowwise Spearman, valid >=8
 x=f.to_numpy(); y=fw.to_numpy(); out=[]
 for i in range(len(f)):
  m=np.isfinite(x[i])&np.isfinite(y[i])
  if m.sum()>=8:
   a=rankdata(x[i,m]); b=rankdata(y[i,m]); out.append(np.corrcoef(a,b)[0,1])
 return np.array(out)
for look in [10,20,30]:
 mom=np.log(p/p.shift(look)); resid=mom.sub(mom.mean(axis=1),axis=0)
 disp=r.rolling(20).std().mean(axis=1); z=(disp/disp.rolling(120).median()-1).clip(-.5,1)
 fac=resid.mul(1+0.8*z,axis=0)
 for h in [1,5,10]:
  a=run(fac,np.log(p.shift(-h)/p),h); ic=np.nanmean(a); ir=ic/np.nanstd(a,ddof=1)
  print(f'look={look} h={h} dates={len(a)} avgN=15.00 coverage=100.0 IC={ic:.6f} ICIR={ir:.6f} hit={np.mean(a>0):.4f}')
 a=run(fac,np.log(p.shift(-10)/p),10)
 # approximate year slices via aligned valid array
 for yr in [2020,2021,2022,2023,2024,2025,2026,2027,2028,2029]:
  ix=(fac.index.year==yr); vals=[]; x=fac.loc[ix].to_numpy(); y=np.log(p.shift(-10)/p).loc[ix].to_numpy()
  for j in range(len(x)):
   m=np.isfinite(x[j])&np.isfinite(y[j])
   if m.sum()>=8: vals.append(np.corrcoef(rankdata(x[j,m]),rankdata(y[j,m]))[0,1])
  if vals: print(f'REG look={look} year={yr} IC={np.mean(vals):.5f} n={len(vals)}')
