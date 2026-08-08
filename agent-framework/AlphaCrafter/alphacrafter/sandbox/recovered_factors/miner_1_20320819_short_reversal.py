import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
d=pd.DataFrame(px).sort_index(); r=d.pct_change()
# single idea: short-term reversal, factor known only at t-1, negative prior 1-day return
f=-r.shift(1)
for h in [1,5,10,20]:
  fr=d.shift(-h)/d-1; vals=[]; ns=[]; dates=[]
  for dt in d.index:
    x=f.loc[dt]; y=fr.loc[dt]
    ok=x.notna()&y.notna()
    if ok.sum()>=8:
      vals.append(spearmanr(x[ok],y[ok]).statistic); ns.append(ok.sum()); dates.append(dt)
  z=np.array(vals); print('H',h,'dates',len(z),'meanN',round(np.mean(ns),2),'IC',round(np.nanmean(z),6),'ICIR',round(np.nanmean(z)/(np.nanstd(z,ddof=1)+1e-12),6),'hit',round(np.mean(z>0),4),'coverage',round(np.mean(ns)/(len(assets)),4))
# rank turnover 10 obs
rank=f.rank(axis=1,pct=True); q=rank.diff(10).abs().mean(axis=1).mean(); print('turnover10',round(q,6))
# regime h1
fr=d.shift(-1)/d-1
for lo,hi in [('2020','2023-12-31'),('2024','2027-12-31'),('2028','2030-12-31'),('2031','2032-08-19')]:
 z=[]
 for dt in d.index:
  if str(dt)[:4]>=lo and str(dt)[:10]<=hi:
   ok=f.loc[dt].notna()&fr.loc[dt].notna()
   if ok.sum()>=8:z.append(spearmanr(f.loc[dt,ok],fr.loc[dt,ok]).statistic)
 z=np.array(z); print('regime',lo,hi,'dates',len(z),'IC',round(np.nanmean(z),6) if len(z) else None,'ICIR',round(np.nanmean(z)/(np.nanstd(z,ddof=1)+1e-12),6) if len(z)>1 else None)
