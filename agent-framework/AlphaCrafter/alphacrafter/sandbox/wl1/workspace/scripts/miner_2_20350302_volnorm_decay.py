import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];px={}
for s in U:
 try:d=get_stock_daily_data(s,days=6000)
 except: d=None
 if d is not None and len(d):px[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(px).sort_index().ffill();r=np.log(p).diff(); f=((r.rolling(20).sum()-r.rolling(60).sum()/3)/r.rolling(20).std()).shift(1)
for h in [15,20,30,40]:
 fw=p.shift(-h)/p-1;z=[]
 for d in f.index:
  x,y=f.loc[d],fw.loc[d];ok=x.notna()&y.notna()
  if ok.sum()>=8:z.append(x[ok].corr(y[ok]))
 z=np.array(z);print(h,len(z),np.nanmean(z),np.nanmean(z)/np.nanstd(z,ddof=1),np.mean(z>0))
