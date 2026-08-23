import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def main():
 Z={};Fs={};dates=set()
 for s in U:
  d=get_stock_daily_data(s,days=1800)
  if d is None or len(d)==0:d=get_index_daily_data(s,days=1800)
  if d is None or len(d)<80:continue
  d=d.copy();d.date=pd.to_datetime(d.date);d=d.set_index('date').sort_index();r=d.close.pct_change();rng=(d.high-d.low).replace(0,np.nan)
  clv=(2*d.close-d.high-d.low)/rng; vol=d.volume.replace(0,np.nan)
  vs=vol/(vol.rolling(30,min_periods=15).median()+1e-12)
  # Recovery quality on down days, confirmed by abnormal volume, with vol capped for robustness
  q=(clv*vs.clip(upper=3)).where(r<0,0.0)
  Z[s]=q.rolling(10,min_periods=10).mean()/(r.rolling(20,min_periods=20).std()+1e-8)
  Fs[s]={h:d.close.shift(-h)/d.close-1 for h in [5,10,20]};dates.update(d.index)
 idx=sorted(dates)
 for h in [5,10,20]:
  ics=[]
  for t in idx:
   a=[];b=[]
   for s,z in Z.items():
    f=Fs[s][h]
    if t in z.index and np.isfinite(z.loc[t]) and np.isfinite(f.loc[t]):a.append(z.loc[t]);b.append(f.loc[t])
   if len(a)>=8 and np.std(a)>0 and np.std(b)>0:ics.append(spearmanr(a,b).statistic)
  x=np.array(ics);print('horizon',h,'dates',len(x),'avg_n',len(Z),'IC',x.mean(),'ICIR',x.mean()/(x.std(ddof=1)+1e-12),'hit',np.mean(x>0),'std',x.std(ddof=1))
if __name__=='__main__':main()
