import pandas as pd, numpy as np, glob, os, json
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')
 # close-location pressure: average close position in daily range, with zero-range handled
 rng=(d.high-d.low).replace(0,np.nan)
 clv=((2*d.close-d.high-d.low)/rng).rolling(10,min_periods=8).mean()
 D[a]=pd.DataFrame({'f':clv,'r':d.close.pct_change()})
idx=sorted(set.intersection(*[set(x.index) for x in D.values()]))
F=pd.DataFrame({a:D[a].f for a in assets}).reindex(idx); R=pd.DataFrame({a:D[a].r for a in assets}).reindex(idx)
print('period',idx[0],idx[-1],'assets',len(assets))
for h in [1,5,10,20]:
  fr=F.shift(1); fw=R.shift(-h).rolling(h).sum().shift(-(h-1)) # return t+1..t+h, aligned t
  vals=[]; ns=[]
  for dt in idx:
   x=fr.loc[dt]; y=fw.loc[dt]; ok=x.notna()&y.notna()
   if ok.sum()>=8: vals.append(spearmanr(x[ok],y[ok]).statistic);ns.append(ok.sum())
  z=np.array(vals); print('H',h,'dates',len(z),'meanN',round(np.mean(ns),2),'IC',round(np.nanmean(z),6),'ICIR',round(np.nanmean(z)/(np.nanstd(z,ddof=1)+1e-12),6),'hit',round(np.mean(z>0),3),'coverage',round(F.notna().mean().mean(),3))
  for lo,hi in [('2020','2024'),('2025','2027'),('2028','2029'),('2030','2031')]:
   q=z[[str(idx[i])[:4]>=lo and str(idx[i])[:4]<=hi for i in range(len(idx))][:len(z)]] if False else None
  # explicit date regime
  for name,mask in [('2020-24',(np.array([d.year for d in idx])<=2024)),('2025-27',np.array([d.year for d in idx]).between if False else np.array([(d.year>=2025 and d.year<=2027) for d in idx])),('2028-29',np.array([(d.year>=2028 and d.year<=2029) for d in idx])),('latest120',np.arange(len(idx))>=len(idx)-120)]:
   # rebuild matching observations
   vv=[]
   for dt in idx[np.asarray(mask)]:
    x=fr.loc[dt];y=fw.loc[dt];ok=x.notna()&y.notna()
    if ok.sum()>=8: vv.append(spearmanr(x[ok],y[ok]).statistic)
   if len(vv): print(' ',name,len(vv),round(np.mean(vv),6),round(np.mean(vv)/(np.std(vv,ddof=1)+1e-12),6))
# audit candidate temporal turnover
print('turnover',round(np.mean((F.rank(axis=1,pct=True).diff().abs().sum(axis=1)/2).dropna()/len(assets)),4))
