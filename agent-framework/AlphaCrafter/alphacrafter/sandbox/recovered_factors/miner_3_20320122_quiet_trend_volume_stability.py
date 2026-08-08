import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
from pathlib import Path
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={Path(f).stem:pd.read_csv(f,parse_dates=['date']).set_index('date') for f in glob.glob('../persistent/stock_data/*.csv')}
A=[a for a in A if a in D]
close=pd.DataFrame({a:D[a].close for a in A}).sort_index().ffill(); vol=pd.DataFrame({a:D[a].volume for a in A}).sort_index().ffill(); r=close.pct_change()
# Quiet-trend persistence: medium horizon return, penalized by unstable volume participation.
# Volume instability is distinct from price volatility and suppresses event-driven trends.
ret=r.rolling(20).sum(); rv=r.rolling(20).std(); v=np.log1p(vol); vinst=v.rolling(20).std()/v.rolling(60).mean()
f=ret/(rv*np.sqrt(20)+1e-12)/(1+vinst.clip(lower=0))
print('assets',len(A),'dates',len(close),'signal_cells',int(f.notna().sum().sum()),'coverage',f.notna().mean().mean())
for h in [1,5,10,20]:
 fw=close.shift(-h)/close-1; z=[]; ns=[]
 for d in f.index:
  ok=f.loc[d].notna()&fw.loc[d].notna()
  if ok.sum()>=8:
   z.append(spearmanr(f.loc[d][ok],fw.loc[d][ok]).statistic); ns.append(ok.sum())
 s=pd.Series(z); recent=s.tail(120)
 print('H',h,'dates',len(s),'meanN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4),'recentIC',round(recent.mean(),6),'recentICIR',round(recent.mean()/recent.std(ddof=1),6))
print('turnover10',f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean())
# regime blocks
fw=close.shift(-10)/close-1
for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2032')]:
 z=[]
 for d in f.index:
  if lo<=str(d.year)<=hi:
   ok=f.loc[d].notna()&fw.loc[d].notna()
   if ok.sum()>=8:z.append(spearmanr(f.loc[d][ok],fw.loc[d][ok]).statistic)
 s=pd.Series(z);print('REG',lo,hi,len(s),s.mean() if len(s) else np.nan,s.mean()/s.std(ddof=1) if len(s)>1 else np.nan)
