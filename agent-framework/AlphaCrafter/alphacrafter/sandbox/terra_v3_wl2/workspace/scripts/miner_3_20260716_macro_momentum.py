import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
p=pd.concat(D,axis=1).sort_index(); r=np.log(p).diff()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(p.index).ffill()
# regime: VIX level relative to 60d median; momentum sign is inverted in stress (defensive cross asset response)
reg=np.where(v>v.rolling(60).median(),-1,1)
f=p.pct_change(20).div(r.rolling(60).std()*np.sqrt(20)).mul(reg,axis=0)
ics=[]; turns=[]; prev=None; ns=[]; dates=[]
for i in range(60,len(p)-1):
 z=pd.concat([f.iloc[i],p.iloc[i+1]/p.iloc[i]-1],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic; ics.append(q);ns.append(len(z)); dates.append(p.index[i])
  if prev is not None: turns.append(np.mean(np.sign(f.iloc[i].reindex(U).fillna(0))!=np.sign(prev.reindex(U).fillna(0))))
  prev=f.iloc[i]
a=np.array(ics); print('dates',len(a),'avg_n',np.mean(ns),'coverage',np.sum(ns)/(len(ns)*15),'ic',a.mean(),'icir',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'turn',np.mean(turns))
for k in [0,250,500]:
 b=a[k:] if k else a
 print('slice',k,'n',len(b),'ic',b.mean(),'icir',b.mean()/b.std(ddof=1))
print('stress',a[np.array([reg[p.index.get_loc(x)]<0 for x in dates])].mean(),'calm',a[np.array([reg[p.index.get_loc(x)]>0 for x in dates])].mean())
