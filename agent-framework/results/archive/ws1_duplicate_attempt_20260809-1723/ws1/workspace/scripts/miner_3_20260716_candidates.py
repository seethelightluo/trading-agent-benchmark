import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-07-15')
x={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 d=d.loc[:cut]
 r=d.close.pct_change()
 x[s]=pd.DataFrame({'close':d.close,'r':r,'mom20':d.close.pct_change(20),'rev5':-d.close.pct_change(5),'ram20':d.close.pct_change(20)/(r.rolling(20).std()+1e-8),'trend':d.close/d.close.rolling(60).mean()-1})
idx=sorted(set.intersection(*[set(v.index) for v in x.values()]))
for f in ['mom20','rev5','ram20','trend']:
  vals=[]
  for dt in idx[:-1]:
    a=[]; b=[]
    nd=idx[idx.index(dt)+1]
    for s in U:
      z=x[s]
      if dt in z.index and nd in z.index and pd.notna(z.loc[dt,f]) and pd.notna(z.loc[nd,'r']): a.append(z.loc[dt,f]); b.append(z.loc[nd,'r'])
    if len(a)>=8: vals.append(spearmanr(a,b).statistic)
  a=np.array(vals); print(f,'dates',len(a),'meanIC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'absmean',np.mean(np.abs(a)))
  # 10d decay
  for h in [5,10]:
   vs=[]
   for i,dt in enumerate(idx[:-h]):
    end=idx[i+h]; aa=[];bb=[]
    for s in U:
     z=x[s]
     if pd.notna(z.loc[dt,f]) and pd.notna(z.loc[end,'close']): aa.append(z.loc[dt,f]);bb.append(z.loc[end,'close']/z.loc[dt,'close']-1)
    if len(aa)>=8:vs.append(spearmanr(aa,bb).statistic)
   print(' ',h,np.nanmean(vs),len(vs))
