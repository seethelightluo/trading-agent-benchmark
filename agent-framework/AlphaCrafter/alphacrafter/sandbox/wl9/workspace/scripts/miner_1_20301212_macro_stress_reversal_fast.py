import os,numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float).sort_index()
px=pd.concat([load(s).rename(s) for s in U],axis=1).sort_index()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].astype(float).sort_index()
r=px.pct_change(10); vol=px.pct_change().rolling(20).std()*np.sqrt(252)
stress=(vix/vix.rolling(60).median()-1).clip(0,2).reindex(px.index).ffill()
f=(-(r/vol)*(1+stress)).shift(1)
for h in [5,10,20,40,60]:
 y=px.pct_change(h).shift(-h); a=f.rank(axis=1); b=y.rank(axis=1)
 ic=a.corrwith(b,axis=1,method='pearson'); n=f.notna().sum(axis=1).where(f.notna().sum(axis=1).eq(y.notna().sum(axis=1)), np.nan)
 # exact joint valid count
 n=(f.notna()&y.notna()).sum(axis=1); ok=ic.notna()&(n>=8)
 z=ic[ok]; print('h',h,'dates',len(z),'avgN',round(n[ok].mean(),2),'coverage',round(f[ok].notna().sum().sum()/(15*len(z)),4),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
 for label,mask in [('2024-26',(z.index>='2024-01-01')&(z.index<'2027-01-01')),('2027-29',(z.index>='2027-01-01')&(z.index<'2030-01-01')),('2030YTD',z.index>='2030-01-01')]:
  q=z[mask]
  if len(q)>5: print(' ',label,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
rank=f.rank(axis=1,pct=True); print('turnover',round(rank.diff().abs().mean(axis=1).mean()/2,6),'valid_dates',f.notna().any(axis=1).sum(),'period',f.index.min().date(),f.index.max().date())
