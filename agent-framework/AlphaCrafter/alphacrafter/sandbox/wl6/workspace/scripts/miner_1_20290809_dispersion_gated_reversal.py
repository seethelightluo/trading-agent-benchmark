import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base=Path('../persistent/stock_data'); cutoff=pd.Timestamp('2029-08-08')
px={s:pd.read_csv(base/(s+'.csv'),parse_dates=['date']).set_index('date')['close'] for s in U}
prices=pd.DataFrame(px).sort_index().loc[:cutoff]; r=prices.pct_change()
ret10=prices.pct_change(10); vol30=r.rolling(30).std()*np.sqrt(252)
disp=r.rolling(10).std().mean(axis=1)
base_sig=-(ret10.sub(ret10.mean(axis=1),axis=0))/(vol30+0.01)
mult=(disp/disp.rolling(120).median()).clip(0.5,2.0)
sig=base_sig.mul(mult,axis=0)
for h in [1,5,10,20]:
 f=sig; fr=prices.shift(-h).div(prices).sub(1); ics=[]; dates=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(d); ns.append(len(z))
 a=np.array(ics); print(h,'dates',len(a),'avgN',np.mean(ns),'coverage',np.mean(ns)/15,'IC %.6f'%np.nanmean(a),'ICIR %.6f'%(np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12)*np.sqrt(len(a))),'hit',np.mean(a>0))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026),(2027,2028),(2029,2029)]:
  q=a[(pd.Series(dates).dt.year>=lo)&(pd.Series(dates).dt.year<=hi)]; print(' ',lo,hi,len(q),round(np.nanmean(q),6) if len(q) else None)
