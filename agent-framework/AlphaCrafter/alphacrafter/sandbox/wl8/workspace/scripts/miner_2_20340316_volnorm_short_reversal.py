import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=6000)
 if d is not None and len(d):
  d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.set_index('date').sort_index(); D[s]=d['close'].astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change();
# short-horizon reversal, normalized by idiosyncratic recent risk and damped in high dispersion
vol=r.rolling(20,min_periods=15).std(); disp=r.sub(r.mean(axis=1),axis=0).rolling(20,min_periods=15).std().mean(axis=1)
raw=-(p.pct_change(5))/vol
# cross-sectional standardize and smooth, with dispersion gate bounded 0.5..1.5
cs=raw.sub(raw.mean(axis=1),axis=0).div(raw.std(axis=1),axis=0)
gate=(disp/disp.rolling(252,min_periods=100).median()).clip(.5,1.5)
f=cs.mul(gate,axis=0).rolling(3,min_periods=3).mean()
fr=p.shift(-10)/p-1
ics=[]; rows=[]
for dt in f.index:
 x=f.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  ics.append(ic); rows.append((dt,ic,len(z)))
ic=np.array(ics); dates=[x[0] for x in rows]
print('dates',len(ic),'avgN',np.mean([x[2] for x in rows]),'start',min(dates),'end',max(dates))
print('IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',np.mean(ic>0),'coverage',np.mean([x[2] for x in rows])/15)
for n in [365,750,1260]:
 a=ic[-n:]; print('recent',n,a.mean(),a.mean()/a.std(ddof=1),len(a))
for h in [1,5,20]:
 yy=p.shift(-h)/p-1; aa=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(aa),len(aa))
# turnover: rank ordering changes
rank=f.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean().mean())
f.to_csv('scripts/miner_2_20340316_volnorm_short_reversal_signal.csv'); pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_2_20340316_volnorm_short_reversal_ic.csv',index=False)
