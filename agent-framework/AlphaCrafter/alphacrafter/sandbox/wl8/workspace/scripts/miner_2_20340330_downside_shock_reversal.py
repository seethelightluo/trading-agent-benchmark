import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,days=6000)
 if d is not None and len(d): d=d.copy(); d.date=pd.to_datetime(d.date); D[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change(); dn=r.where(r<0,0).rolling(20,min_periods=15).std()
# shock reversal: recent 3d loss reversal, scaled by downside risk and damped for persistent trends
shock=-p.pct_change(3)/(dn+1e-12)
trend=p.pct_change(20)
raw=shock*(1-(trend>0).astype(float)*.25)
f=raw.sub(raw.mean(axis=1),axis=0).div(raw.std(axis=1),axis=0).rolling(2,min_periods=2).mean()
fr=p.shift(-10)/p-1; rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
a=np.array([x[1] for x in rows]); dates=[x[0] for x in rows]
print('dates',len(a),'avgN',np.mean([x[2] for x in rows]),'start',min(dates),'end',max(dates)); print('IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'coverage',np.mean([x[2] for x in rows])/15)
for n in [365,750,1260]: q=a[-n:]; print('recent',n,q.mean(),q.mean()/q.std(ddof=1),len(q))
for h in [1,5,20]:
 yy=p.shift(-h)/p-1;q=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.mean(q),len(q))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
f.to_csv('scripts/miner_2_20340330_downside_shock_reversal_signal.csv');pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_2_20340330_downside_shock_reversal_ic.csv',index=False)
