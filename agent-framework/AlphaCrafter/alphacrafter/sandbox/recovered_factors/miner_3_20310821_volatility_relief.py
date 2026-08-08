import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
from pathlib import Path
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
files=glob.glob('../persistent/stock_data/*.csv')
px=pd.DataFrame({Path(f).stem:pd.read_csv(f,parse_dates=['date']).set_index('date')['close'] for f in files}).sort_index().ffill()
px=px[[a for a in A if a in px]]; r=px.pct_change()
# Volatility relief: assets whose recent realized risk is falling relative to their own medium-term risk.
v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std()
f=-(v20/(v60+1e-12)-1)
# smooth to reduce noise, with signal observable at t only
f=f.rolling(5,min_periods=3).mean()
print('instruments',px.shape[1],'rows',len(px),'period',px.index.min(),px.index.max())
for h in [1,5,10,20]:
 fw=px.shift(-h)/px-1; z=[]; ns=[]; ds=[]
 for d in f.index:
  ok=f.loc[d].notna()&fw.loc[d].notna()
  if ok.sum()>=8:
   z.append(spearmanr(f.loc[d][ok],fw.loc[d][ok]).statistic);ns.append(ok.sum());ds.append(d)
 z=pd.Series(z,index=ds)
 print('H',h,'dates',len(z),'meanN',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4),'latest120',round(z.tail(120).mean(),6),round(z.tail(120).mean()/z.tail(120).std(ddof=1),6))
print('coverage',round(f.notna().mean().mean(),4),'turnover10',round(f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean(),4))
fw=px.shift(-10)/px-1
for lab,mask in [('2020-23',f.index<'2024'),('2024-27',(f.index>='2024')&(f.index<'2028')),('2028-30',(f.index>='2028')&(f.index<'2031')),('2031',f.index>='2031')]:
 q=[]
 for d in f.index[mask]:
  ok=f.loc[d].notna()&fw.loc[d].notna()
  if ok.sum()>=8:q.append(spearmanr(f.loc[d][ok],fw.loc[d][ok]).statistic)
 print(lab,'dates',len(q),'IC',round(np.mean(q),6) if q else None,'ICIR',round(np.mean(q)/np.std(q,ddof=1),6) if len(q)>1 else None)
