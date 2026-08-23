import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def fetch(s):
 d=get_stock_daily_data(s,days=1800)
 if d is None or len(d)==0: d=get_index_daily_data(s,days=1800)
 return d

def main():
 sigs={}; futs={}; allidx=[]
 for s in U:
  d=fetch(s)
  if d is None: continue
  d=d.copy(); d.date=pd.to_datetime(d.date); d=d.set_index('date').sort_index()
  r=d.close.pct_change(); rng=(d.high-d.low).replace(0,np.nan)
  clv=(2*d.close-d.high-d.low)/rng
  rec=(d.close-d.low)/((d.open-d.low).replace(0,np.nan))
  q=(.65*clv+.35*(2*rec-1)).where(r<0,.25*clv)
  sigs[s]=q.rolling(10,min_periods=10).mean()/(r.rolling(20,min_periods=20).std()+1e-8)
  futs[s]=d.close.shift(-10)/d.close-1; allidx += list(d.index)
 idx=sorted(set(allidx)); out=[]
 for dt in idx:
  a=[];b=[]
  for s in sigs:
   if dt in sigs[s].index and np.isfinite(sigs[s].loc[dt]) and np.isfinite(futs[s].loc[dt]): a.append(sigs[s].loc[dt]);b.append(futs[s].loc[dt])
  if len(a)>=8 and np.std(a)>0 and np.std(b)>0: out.append(spearmanr(a,b).statistic)
 x=np.array(out); print('factor=intraday_recovery_quality_10d dates',len(x),'avg_n~',15,'coverage~',len(x)/len(idx),'IC',x.mean(),'ICIR',x.mean()/(x.std(ddof=1)+1e-12),'hit',np.mean(x>0),'std',x.std(ddof=1))
 for h in [5,20]:
  vals=[]
  for s in U:
   d=fetch(s)
   if d is None: continue
   d=d.copy();d.date=pd.to_datetime(d.date);d=d.set_index('date').sort_index();r=d.close.pct_change();rng=(d.high-d.low).replace(0,np.nan);clv=(2*d.close-d.high-d.low)/rng;rec=(d.close-d.low)/((d.open-d.low).replace(0,np.nan));q=(.65*clv+.35*(2*rec-1)).where(r<0,.25*clv); z=q.rolling(10,min_periods=10).mean()/(r.rolling(20,min_periods=20).std()+1e-8); vals.append((z,d.close.shift(-h)/d.close-1))
  rr=[]
  for dt in idx:
   aa=[];bb=[]
   for z,f in vals:
    if dt in z.index and np.isfinite(z.loc[dt]) and np.isfinite(f.loc[dt]):aa.append(z.loc[dt]);bb.append(f.loc[dt])
   if len(aa)>=8:rr.append(spearmanr(aa,bb).statistic)
  print('decay',h,'IC',np.nanmean(rr),'dates',len(rr))
if __name__=='__main__':main()
