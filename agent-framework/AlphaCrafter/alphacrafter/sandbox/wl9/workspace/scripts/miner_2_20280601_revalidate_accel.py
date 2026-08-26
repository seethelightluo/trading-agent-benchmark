import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];end=pd.Timestamp('2028-05-31')
px=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index().close.loc[:end] for s in U}).sort_index();r=px.pct_change(); fac=(px/px.shift(5)-1).rank(axis=1,pct=True)-(px/px.shift(20)-1).rank(axis=1,pct=True);fac=fac/r.rolling(20,min_periods=15).std()
def run(h):
 f=px.shift(-h)/px-1;a=[];ds=[];ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 a=np.array(a);ds=pd.to_datetime(ds); return a,ds,np.mean(ns)
def S(a):return len(a),float(np.mean(a)),float(np.mean(a)/(np.std(a,ddof=1)/np.sqrt(len(a)))),float(np.mean(a>0))
for h in [1,5,10,20]:
 a,d,n=run(h);print(h,S(a),'avgN',n,'recent252',S(a[-252:]),'online',S(a[d>=pd.Timestamp('2026-07-16')]))
print('coverage',float(fac.notna().mean().mean()),'turnover',float(fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()*2))
