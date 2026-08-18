import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
p=pd.concat(D,axis=1).sort_index()
r=np.log(p).diff()
# acceleration: intermediate trend (20d) less long trend (60d), normalized by 60d volatility
f=(p.pct_change(20)-p.pct_change(60))/(r.rolling(60).std()*np.sqrt(60))
# forward one and 5/10 day returns
ics={h:[] for h in [1,5,10]}; turns=[]; dates=0; valid_sum=0; prev=None
for i in range(60,len(p)-10):
 x=f.iloc[i]; y={h:p.iloc[i+h]/p.iloc[i]-1 for h in ics}
 for h in ics:
  z=pd.concat([x,y[h]],axis=1).dropna()
  if len(z)>=8: ics[h].append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 z=x.dropna()
 if len(z)>=8:
  dates+=1;valid_sum+=len(z)
  if prev is not None: turns.append(np.mean(np.sign(z.reindex(U).fillna(0))!=np.sign(prev.reindex(U).fillna(0))))
  prev=x
for h,a in ics.items():
 a=np.array(a); print(h,'n',len(a),'ic',a.mean(),'icir',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
print('dates',dates,'avg_n',valid_sum/max(dates,1),'turn',np.mean(turns),'coverage',valid_sum/(dates*15))
# recent and regime halves
for label,a in [('early',np.array(ics[5])[:len(ics[5])//2]),('late',np.array(ics[5])[len(ics[5])//2:]),('recent250',np.array(ics[5])[-250:])]: print(label,a.mean(),a.mean()/a.std(ddof=1),len(a))
# cross-sectional signal correlation to existing concepts
print('factor cross-sectional time correlation with simple 5d rev',f.stack().corr((-p.pct_change(5)).stack()))
