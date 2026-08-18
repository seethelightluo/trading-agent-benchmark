import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2032-08-19')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].loc[:CUT] for s in U}
p=pd.DataFrame(D).sort_index(); r=p.pct_change(); resid=r.sub(r.mean(axis=1),axis=0)
resid20=resid.rolling(20,min_periods=15).sum(); down60=r.clip(upper=0).pow(2).rolling(60,min_periods=40).mean().pow(.5)*np.sqrt(252)
csmed=down60.median(axis=1); rel=(down60.div(csmed,axis=0)).clip(.5,2)
sig=(-(resid20/down60.replace(0,np.nan))*rel).shift(1)
for h in [10,20,30]:
 f=p.shift(-h)/p-1; vals=[]; ns=[]; turns=[]; prev=None
 for d in sig.index:
  q=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(q)>=8:
   vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); ns.append(len(q)); rk=q.iloc[:,0].rank(pct=True)
   if prev is not None: turns.append(np.mean(abs(rk-prev)))
   prev=rk
 a=np.array(vals); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12),6),'hit',round(np.mean(a>0),4),'coverage',round(np.mean(ns)/15,4),'turnover',round(np.nanmean(turns),6))
print('last',p.index.max().date())
