import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
from pathlib import Path
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={Path(f).stem:pd.read_csv(f,parse_dates=['date']).set_index('date') for f in glob.glob('../persistent/stock_data/*.csv')}; A=[a for a in A if a in D]
c=pd.DataFrame({a:D[a].close for a in A}).sort_index().ffill(); r=c.pct_change(); med=r.median(axis=1)
# Relative trend persistence: asset return in excess of contemporaneous cross-asset median,
# accumulated over 40 sessions and volatility-normalized; isolates idiosyncratic leadership.
ex=r.sub(med,axis=0); f=ex.rolling(40).sum()/(r.rolling(40).std()*np.sqrt(40)+1e-12)
print('assets',len(A),'dates',len(c),'signal_cells',int(f.notna().sum().sum()),'coverage',f.notna().mean().mean())
for h in [1,5,10,20]:
 fw=c.shift(-h)/c-1; z=[];ns=[]
 for d in f.index:
  ok=f.loc[d].notna()&fw.loc[d].notna()
  if ok.sum()>=8:z.append(spearmanr(f.loc[d][ok],fw.loc[d][ok]).statistic);ns.append(ok.sum())
 s=pd.Series(z); print('H',h,'dates',len(s),'meanN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4),'recent120',round(s.tail(120).mean(),6),round(s.tail(120).mean()/s.tail(120).std(ddof=1),6))
print('turnover10',f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean())
