import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
 x=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date')
 D[a]=x.close.astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
# lagged medium-term continuation, scaled by realized volatility; all inputs through t-1
mom=p.shift(1)/p.shift(1).shift(20)-1
vol=r.shift(1).rolling(40,min_periods=25).std()*np.sqrt(20)
f=(mom/(0.01+vol)).clip(-20,20)
for h in [5,10,20]:
 fr=p.shift(-h)/p-1; ics=[]; ns=[]; turns=[]; prev=None
 for dt in p.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
  rk=f.loc[dt].rank(pct=True)
  if prev is not None:
   q=pd.concat([rk,prev],axis=1).dropna(); turns.append(np.mean(abs(q.iloc[:,0]-q.iloc[:,1])))
  prev=rk
 ic=np.array(ics); print({'horizon':h,'dates':len(ic),'avg_n':round(np.mean(ns),2),'coverage':round(np.mean(ns)/15,4),'ic':round(np.mean(ic),6),'icir':round(np.mean(ic)/np.std(ic,ddof=1),4),'hit':round(np.mean(ic>0),4),'turnover':round(np.nanmean(turns),4)})
print('data_end',p.index.max().date())
