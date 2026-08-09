import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
from pathlib import Path
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
files=glob.glob('../persistent/stock_data/*.csv');px=pd.DataFrame({Path(f).stem:pd.read_csv(f,parse_dates=['date']).set_index('date')['close'] for f in files}).sort_index().ffill();px=px[[a for a in A if a in px]];r=px.pct_change(); f=px.pct_change(40)/(np.sqrt(r.clip(upper=0).pow(2).rolling(60,min_periods=40).mean())*np.sqrt(252)+1e-8); f=f.clip(lower=f.quantile(.05,axis=1),upper=f.quantile(.95,axis=1),axis=0)
print('instruments',len(f.columns),'rows',len(f),'coverage',round(f.notna().mean().mean(),4))
for h in [10,20,40]:
 fw=px.shift(-h)/px-1;z=[];ds=[];ns=[]
 for d in f.index:
  ok=f.loc[d].notna()&fw.loc[d].notna()
  if ok.sum()>=8:
   q=spearmanr(f.loc[d][ok],fw.loc[d][ok]).statistic
   if np.isfinite(q):z.append(q);ds.append(d);ns.append(ok.sum())
 z=pd.Series(z,index=ds);print('H',h,'dates',len(z),'meanN',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'latest120',round(z.tail(120).mean(),6),round(z.tail(120).mean()/z.tail(120).std(ddof=1),6))
print('turnover10',round(f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean(),4))
