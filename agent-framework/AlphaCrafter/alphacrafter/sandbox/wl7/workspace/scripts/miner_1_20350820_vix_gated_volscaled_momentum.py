import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; px={}
for a in assets:
 d=pd.read_csv(f'{base}/{a}.csv',parse_dates=['date']).set_index('date')['close'].astype(float); px[a]=d
p=pd.DataFrame(px).sort_index(); macro=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].astype(float).reindex(p.index).ffill(); r=p.pct_change()
mom=p.pct_change(20); vol=r.rolling(40).std()*np.sqrt(252); vr=macro.rolling(60,min_periods=30).rank(pct=True)
f=mom.div(vol).mul(np.where(vr.values[:,None]>.70,-0.5,1.0),axis=0)
for h in [5,10,20]:
 fr=p.shift(-h).div(p)-1; ics=[]; cov=[]; ns=[]
 for dt in p.index:
  x=f.loc[dt]; y=fr.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8:
   ics.append(spearmanr(x[ok],y[ok]).statistic); cov.append(ok.mean()); ns.append(ok.sum())
 z=pd.Series(ics).dropna(); turns=[]
 for i in range(1,len(p)):
  a=f.iloc[i].rank(pct=True); b=f.iloc[i-1].rank(pct=True); turns.append((a-b).abs().mean())
 print('H',h,'obs',len(z),'avgN',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4),'coverage',round(np.mean(cov),4),'turnover',round(np.mean(turns),4),'period',p.index.min().date(),p.index.max().date())
