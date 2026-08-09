import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-16')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut] for s in U}
p=pd.concat(D,axis=1).sort_index(); r=p.pct_change()
# Common-factor-neutral momentum: trailing return minus trailing cross-sectional median return,
# using only completed observations. Test 20d and 60d windows.
for w in [20,40,60,120]:
 ret=p.pct_change(w)
 common=ret.median(axis=1)
 f=ret.sub(common,axis=0)
 vals=[]; ns=[]; dates=[]
 for i in range(len(r)-1):
  q=pd.concat([f.iloc[i],r.iloc[i+1].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.y.nunique()>1:
   vals.append(spearmanr(q.iloc[:,0],q.y).statistic); ns.append(len(q)); dates.append(r.index[i])
 a=np.asarray(vals); icir=a.mean()/a.std(ddof=1)
 # rank turnover only dates with usable signal
 rr=f.rank(axis=1,pct=True).loc[dates]
 turnover=np.nanmean(np.abs(rr.diff()).mean(axis=1))
 print({'window':w,'dates':len(a),'avg_names':round(float(np.mean(ns)),2),'ic':round(float(a.mean()),6),'icir':round(float(icir),6),'hit':round(float(np.mean(a>0)),4),'turnover':round(float(turnover),4),'start':str(dates[0].date()),'end':str(dates[-1].date())})
 # regime means
 for yr in [2020,2021,2022,2023,2024,2025,2026]:
  z=a[[d.year==yr for d in dates]]
  if len(z): print(' regime',yr,len(z),round(float(z.mean()),5))
