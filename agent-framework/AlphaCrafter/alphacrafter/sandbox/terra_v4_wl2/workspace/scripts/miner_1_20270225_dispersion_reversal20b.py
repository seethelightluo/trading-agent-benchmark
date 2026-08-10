import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U},axis=1).sort_index().loc[:'2027-02-24']
r=P.pct_change(); m=r.rolling(20,min_periods=15).sum().shift(1); med=m.median(axis=1)
# standard cross-sectional reversal, continuously shrink by dispersion to avoid unstable extreme scaling
for scale in [None,20,60]:
 f=-(m.sub(med,axis=0));
 if scale: f=f/(r.rolling(scale,min_periods=scale).std().median(axis=1).replace(0,np.nan)).shift(1)
 y=P.shift(-5).div(P)-1; vals=[]; ds=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(d);ns.append(len(z))
 v=np.array(vals); print('scale',scale,'dates',len(v),'N',np.mean(ns),'IC',np.mean(v),'ICIR',np.mean(v)/np.std(v,ddof=1)*np.sqrt(len(v)),'hit',np.mean(v>0))
 for yr in ['2026','2027']:
  q=v[[str(d)[:4]==yr for d in ds]]; print(yr,len(q),np.mean(q) if len(q) else None)
