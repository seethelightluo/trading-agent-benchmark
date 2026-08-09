import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-30')
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]
 D[s]=x[['close','volume']]
p=pd.concat({s:D[s].close for s in U},axis=1).sort_index(); vol=pd.concat({s:D[s].volume for s in U},axis=1).reindex(p.index)
r=p.pct_change()
# Volume-confirmed reversal: fade recent return, scaled by abnormal log-volume.
for h,win,alpha in [(2,20,0.5),(3,20,0.5),(5,20,0.5),(3,60,0.5),(3,20,1.0)]:
 z=np.log(vol.replace(0,np.nan)).sub(np.log(vol.replace(0,np.nan)).rolling(win).mean())
 f=-p.pct_change(h)*(1+alpha*z.clip(-2,2))
 vals=[]; ns=[]; ds=[]
 for i in range(len(r)-1):
  q=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   vals.append(spearmanr(q.f,q.y).statistic); ns.append(len(q)); ds.append(r.index[i])
 a=np.asarray(vals); dates=pd.DatetimeIndex(ds)
 print('CFG',h,win,alpha,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4),'years',[(y,round(a[dates.year==y].mean(),5),int((dates.year==y).sum())) for y in range(2020,2027) if (dates.year==y).any()])
 print('decay',[(k,round(np.nanmean([spearmanr(pd.concat([f.iloc[i],r.shift(-k).iloc[i]],axis=1).dropna().iloc[:,0],pd.concat([f.iloc[i],r.shift(-k).iloc[i]],axis=1).dropna().iloc[:,1]).statistic for i in range(len(r)-k) if len(pd.concat([f.iloc[i],r.shift(-k).iloc[i]],axis=1).dropna())>=8]),5)) for k in [1,5,10]])
