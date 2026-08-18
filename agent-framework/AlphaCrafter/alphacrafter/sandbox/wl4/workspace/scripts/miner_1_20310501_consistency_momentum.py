import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 try:D[s]=get_index_daily_data(s,4000)
 except:D[s]=get_stock_daily_data(s,4000)
px=pd.concat({s:d.set_index('date').close for s,d in D.items()},axis=1).sort_index().ffill(); r=px.pct_change(); vol=r.rolling(20).std(); ret=px.pct_change(20); consistency=(r>0).rolling(40).mean()
# Consistency-adjusted, volatility-normalized intermediate momentum, lagged one day.
f=(ret*(.25+consistency)/(vol*np.sqrt(20))).shift(1)
def test(h,start=0):
 a=[]; ns=[]
 for i in range(start,len(px)-h):
  y=px.iloc[i+h]/px.iloc[i]-1; x=f.iloc[i]; ok=x.notna()&y.notna();ns.append(ok.sum())
  if ok.sum()>=8:a.append(spearmanr(x[ok],y[ok]).statistic)
 a=np.asarray(a);return len(a),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean(),np.mean(ns)
print('dates',len(px),'assets',len(U),'coverage',f.notna().mean().mean())
for h in [5,10,20]:print('H',h,test(h))
for n in [365,730,1095]:print('recent',n,test(10,max(0,len(px)-n-10)))
print('turnover',np.nanmean(f.rank(pct=True).diff().abs().mean(axis=1)))
