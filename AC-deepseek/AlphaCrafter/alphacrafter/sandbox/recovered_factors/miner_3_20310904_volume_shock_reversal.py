import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
from pathlib import Path
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={Path(f).stem:pd.read_csv(f,parse_dates=['date']).set_index('date') for f in glob.glob('../persistent/stock_data/*.csv')}; common=[a for a in A if a in D]; close=pd.DataFrame({a:D[a].close for a in common}).sort_index().ffill(); vol=pd.DataFrame({a:D[a].volume for a in common}).sort_index().ffill(); r=close.pct_change()
# Volume-confirmed reversal: recent negative return on unusually high volume, expecting mean reversion
shock=(r.rolling(5).sum())*(np.log1p(vol).rolling(20).mean()/np.log1p(vol).rolling(60).mean()-1)
f=-shock
print('assets',len(close.columns),'dates',len(close),'coverage',f.notna().mean().mean())
for h in [1,5,10,20]:
 fw=close.shift(-h)/close-1; z=[];ns=[]
 for d in f.index:
  ok=f.loc[d].notna()&fw.loc[d].notna()
  if ok.sum()>=8:z.append(spearmanr(f.loc[d][ok],fw.loc[d][ok]).statistic);ns.append(ok.sum())
 s=pd.Series(z);print('H',h,'dates',len(s),'meanN',np.mean(ns),'IC',s.mean(),'ICIR',s.mean()/s.std(ddof=1),'hit',(s>0).mean(),'latest120',s.tail(120).mean(),s.tail(120).mean()/s.tail(120).std(ddof=1))
print('turnover10',f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean())
