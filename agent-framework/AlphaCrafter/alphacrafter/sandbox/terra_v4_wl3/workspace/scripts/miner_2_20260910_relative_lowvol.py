import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base=Path('../persistent/stock_data')
def load(s):
 d=pd.read_csv(base/(s+'.csv')); d.date=pd.to_datetime(d.date); return d.drop_duplicates('date').set_index('date').close.astype(float).sort_index()
px=pd.DataFrame({s:load(s) for s in U}).sort_index().loc[:'2026-09-09']; r=px.pct_change(fill_method=None)
# Relative low-volatility: negative asset vol divided by equal-weight universe vol.
univ=r.mean(axis=1); f=-r.rolling(20,min_periods=20).std().div(univ.rolling(20,min_periods=20).std(),axis=0)
def calc(h):
 y=px.pct_change(h,fill_method=None).shift(-h); a=[];ds=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):a.append(q);ds.append(d);ns.append(len(z))
 return pd.Series(a,index=ds),ns
print('candidate=relative_low_vol_20d');print('assets',len(U),'rows',len(f))
for h in [1,5,10]:
 a,n=calc(h);print('h',h,'dates',len(a),'avg_names',round(np.mean(n),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
a,n=calc(1);print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4));print('year',a.groupby(a.index.year).agg(['mean','count']).round(5).to_dict())
