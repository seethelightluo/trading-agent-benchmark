import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,3000); d=d.copy();d.date=pd.to_datetime(d.date);px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index();r=P.pct_change(fill_method=None); vol=r.rolling(20,min_periods=15).std();f=-vol.div(vol.median(axis=1),axis=0);fr=P.pct_change(fill_method=None).shift(-1)
print(P.shape, f.notna().mean().mean(), fr.notna().mean().mean(), f.index.min(),f.index.max())
print(pd.concat([f,fr],axis=1).dropna().shape)
# use intersection per date explicitly
ics=[]; ns=[]
for dt in f.index:
 z=pd.DataFrame({'x':f.loc[dt],'y':fr.loc[dt]}).dropna()
 if len(z)>=8: ics.append(z.x.corr(z.y,method='spearman'));ns.append(len(z))
ic=pd.Series(ics).dropna(); print(len(ic),np.mean(ns),ic.mean(),ic.mean()/ic.std())
