import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
 p=os.path.join(base,s+'.csv')
 if os.path.exists(p):
  d=pd.read_csv(p,parse_dates=['date']).set_index('date')['close'].sort_index(); px[s]=d
P=pd.DataFrame(px).loc[:'2028-11-16']; R=P.pct_change()
# risk-adjusted, consistency-weighted medium trend; lagged naturally by forward return pairing
mom=P.pct_change(20); breadth=(R>0).rolling(20).mean(); vol=R.rolling(20).std()
F=(mom*breadth)/(vol*np.sqrt(20)+1e-12)
ics=[]; rows=[]
for dt in P.index[:-1]:
 x=F.loc[dt]; y=R.shift(-1).loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  ics.append(ic); rows.append((dt,len(z),ic))
a=np.array(ics); print('dates',len(a),'avg_names',np.mean([r[1] for r in rows]),'coverage',np.mean([r[1]/15 for r in rows]))
print('1d IC %.6f ICIR %.6f hit %.4f turnover %.4f'%(np.mean(a),np.mean(a)/(np.std(a,ddof=1)+1e-12),np.mean(a>0),np.nanmean(np.abs(F.rank(axis=1,pct=True).diff()).mean(axis=1))))
for n in [250,500]:
 q=a[-n:]; print('recent',n,'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/(q.std(ddof=1)+1e-12)))
for h in [5,10,20]:
 y=P.pct_change(h).shift(-h)
 aa=[]
 for dt in P.index:
  z=pd.concat([F.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(aa); print('%dd IC %.6f ICIR %.6f dates %d'%(h,q.mean(),q.mean()/(q.std(ddof=1)+1e-12),len(q)))
