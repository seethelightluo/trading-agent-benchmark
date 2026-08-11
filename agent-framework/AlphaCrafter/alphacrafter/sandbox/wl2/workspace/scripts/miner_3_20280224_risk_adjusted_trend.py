import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
SYMS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for s in SYMS:
 d=get_stock_daily_data(s,days=2400)
 if d is None or len(d)<150: d=get_index_daily_data(s,days=2400)
 if d is not None and len(d)>=150:
  d=d.copy();d.date=pd.to_datetime(d.date);raw[s]=d.set_index('date').sort_index().close.astype(float)
p=pd.DataFrame(raw).sort_index().ffill(); ret=np.log(p).diff(); vol=ret.rolling(20,min_periods=20).std(); f=(np.log(p).diff(60)/vol).shift(1)
fr=np.log(p).shift(-1)-np.log(p)
def calc(F,Y,sl=slice(None)):
 a=[]; ns=[]
 for dt in F.loc[sl].index:
  z=pd.concat([F.loc[dt],Y.loc[dt]],axis=1).dropna();
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(a);return len(a),np.mean(ns),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0)
for h in [1,3,5,10]:print('h',h,calc(f,np.log(p).shift(-h)-np.log(p)))
ranks=f.rank(pct=True,axis=1);print('symbols',len(p.columns),'dates',len(p),'coverage',f.notna().sum(axis=1).mean()/len(SYMS),'turnover',ranks.diff().abs().mean(axis=1).mean())
for name,sl in [('2020-22',slice('2020','2022')),('2023-25',slice('2023','2025')),('2026-27',slice('2026','2027')),('2028',slice('2028',None))]:print(name,calc(f,fr,sl))
