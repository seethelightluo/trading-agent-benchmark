import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
from pathlib import Path
files=glob.glob('../persistent/stock_data/*.csv')
px=pd.DataFrame({Path(f).stem:pd.read_csv(f,parse_dates=['date']).set_index('date')['close'] for f in files}).sort_index().ffill()
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px=px[[a for a in assets if a in px]]
r=px.pct_change(); m=r.mean(axis=1)
# upside capture: asset return relative to equal-weight market, only sessions market is positive;
# trailing 60-session average, volatility scaled. Uses only data through t.
rel=r.sub(m,axis=0); f=rel.where(m>0).rolling(60,min_periods=12).mean()/r.rolling(20,min_periods=15).std()
for h in [1,5,10,20]:
 fw=px.shift(-h)/px-1; z=[]; ds=[]; ns=[]
 for d in f.index:
  ok=f.loc[d].notna()&fw.loc[d].notna()
  if ok.sum()>=8: z.append(spearmanr(f.loc[d][ok],fw.loc[d][ok]).statistic);ds.append(d);ns.append(ok.sum())
 z=pd.Series(z,index=ds).dropna();print('H',h,'dates',len(z),'N',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4),'latest120',round(z.tail(120).mean(),6),round(z.tail(120).mean()/z.tail(120).std(ddof=1),6))
print('turnover10',round(f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean(),6),'coverage',round(f.notna().mean().mean(),6),'assets',len(px.columns),'dates',len(px))
for n,s in {'mom20':px.pct_change(20),'invvol20':-r.rolling(20).std(),'downcap':r.where(m<0).rolling(60,min_periods=12).mean()/r.rolling(20).std()}.items():
 q=[]
 for d in f.index:
  ok=f.loc[d].notna()&s.loc[d].notna()
  if ok.sum()>=8:q.append(spearmanr(f.loc[d][ok],s.loc[d][ok]).statistic)
 print('proxy_corr',n,round(np.nanmean(q),6))
