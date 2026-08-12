import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut='2026-07-15'; fs={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').sort_index(); r=x.close.pct_change(); fs[s]=pd.DataFrame({'r':r,'f':r.rolling(20,min_periods=15).sum()/(r.rolling(20,min_periods=15).std()*np.sqrt(20)+1e-12),'y':x.close.shift(-1)/x.close-1})
allr=pd.concat({s:v.r for s,v in fs.items()},axis=1); breadth=allr.mean(axis=1).rolling(5,min_periods=3).mean()
rows=[]
for s,v in fs.items():
 f=v.f.where(breadth>0,-v.f);rows.append(pd.DataFrame({'date':f.index,'s':s,'f':f.values,'y':v.y.values}))
a=pd.concat(rows,ignore_index=True).dropna(); out=[]; ns=[]
for d,g in a.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:
  c=spearmanr(g.f,g.y).statistic
  if pd.notna(c):out.append(c);ns.append(len(g))
q=pd.Series(out);print('conditional_breadth_trend dates',len(q),'avg_n',np.mean(ns),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'coverage',len(a)/(sum(len(v) for v in fs.values())) )
for h in [3,5,10]:
 rows=[]
 for s,v in fs.items(): rows.append(pd.DataFrame({'date':v.index,'f':v.f.where(breadth>0,-v.f).values,'y':v['y'].shift(-(h-1)).values}))
 b=pd.concat(rows,ignore_index=True).dropna(); vv=[]
 for d,g in b.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1:vv.append(spearmanr(g.f,g.y).statistic)
 print('decay',h,len(vv),np.mean(vv),np.mean(vv)/np.std(vv,ddof=1))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-15')]:
 # omit detailed date labels due series index
 pass
