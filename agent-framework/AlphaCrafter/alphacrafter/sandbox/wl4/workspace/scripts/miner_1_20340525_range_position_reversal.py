import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; ds={}
for s in U:
 p=os.path.join(base,s+'.csv')
 if os.path.exists(p):
  d=pd.read_csv(p); d['date']=pd.to_datetime(d['date']); ds[s]=d.set_index('date')['close'].sort_index()
p=pd.DataFrame(ds).sort_index().loc[:'2034-05-25']
# 20d range location: contrarian low location, lag one day
hi=p.rolling(20,min_periods=20).max(); lo=p.rolling(20,min_periods=20).min()
f=-(p-lo)/(hi-lo).replace(0,np.nan)
f=f.shift(1)
fr=p.shift(-10)/p-1
ics=[]; rows=[]
for dt in f.index:
 x=f.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  ics.append(ic); rows.append((dt,ic,len(z)))
a=np.array(ics); print('dates',len(a),'avgN',np.mean([r[2] for r in rows]),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'coverage',len(a)/len(p))
for n in [120,260,520,780]:
 q=a[-n:] if len(a)>=n else a
 print('recent',n,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',np.mean(q>0))
# turnover rank/signal sign changes approximated mean abs change cross-section normalized
r=f.rank(axis=1,pct=True); turn=(r-r.shift(1)).abs().mean(axis=1).dropna(); print('turnover',turn.mean())
# artifacts
os.makedirs('scripts/artifacts',exist_ok=True)
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/artifacts/miner_1_20340525_range_position_reversal_ic.csv',index=False)
f.to_csv('scripts/artifacts/miner_1_20340525_range_position_reversal_signal.csv')
