import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index(); r=p.pct_change(); rr=r.rolling(3).sum(); vol=r.rolling(20).std(); f=-rr.sub(rr.median(axis=1),axis=0)/vol; b=(r>0).sum(axis=1)/r.notna().sum(axis=1)
for cut in [.26,.27,.28,.29,.30,.31,.32,.33]:
 for h in [4,5,6,7]:
  sig=f.where((b<cut)|(b>1-cut)); fr=r.rolling(h).sum(); a=[]
  for i in range(40,len(p)-h):
   z=pd.concat([sig.iloc[i-1],fr.iloc[i+h-1]],axis=1).dropna()
   if len(z)>=8:
    q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
    if np.isfinite(q):a.append(q)
  a=np.array(a)
  print(cut,h,len(a),round(a.mean(),5),round(a.mean()/a.std(ddof=1),5))
