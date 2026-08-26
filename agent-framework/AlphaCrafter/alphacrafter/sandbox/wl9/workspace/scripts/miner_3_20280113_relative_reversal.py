import pandas as pd,numpy as np
from scipy.stats import spearmanr
from pathlib import Path
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2026-07-15'); P={}
for s in S:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index(); P[s]=d.close.loc[:end]
px=pd.DataFrame(P).sort_index(); r=px.pct_change()
# market-relative 3-session reversal: remove cross-asset average move before ranking
raw=px.pct_change(3); fac=-(raw.sub(raw.mean(axis=1),axis=0))
fw=px.shift(-1)/px-1
out=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fw.loc[dt]],axis=1).dropna()
 if len(z)>=8: out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
out=np.array(out); print('dates',len(out),'assets',len(S),'mean_ic',out.mean(),'icir',out.mean()/(out.std(ddof=1)/np.sqrt(len(out))),'hit',(out>0).mean())
print('coverage',fac.notna().mean().mean(),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for h in [1,5,10]:
 fw=px.shift(-h)/px-1; a=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(a);print('decay',h,len(a),a.mean(),a.mean()/(a.std(ddof=1)/np.sqrt(len(a))))
