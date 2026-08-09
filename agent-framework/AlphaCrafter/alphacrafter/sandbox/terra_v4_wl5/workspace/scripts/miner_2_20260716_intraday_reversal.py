import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
data={}
for s in U:
 p=os.path.join(base,s+'.csv'); d=pd.read_csv(p); d['date']=pd.to_datetime(d['date']); d=d.sort_values('date').set_index('date'); data[s]=d
# intraday reversal: prior day's close-to-open move, with range normalization to avoid price scale
# factor positive when close/open was weak (reversal), additionally normalized by rolling 20d intraday volatility
for mode in ['raw','range_scaled']:
 rows=[]
 for s,d in data.items():
  intr=d['close']/d['open']-1
  if mode=='raw': f=-intr
  else: f=-intr/(intr.rolling(20).std()+1e-8)
  fr=d['close'].shift(-1)/d['close']-1
  for dt in d.index:
   rows.append((dt,s,f.loc[dt],fr.loc[dt]))
 x=pd.DataFrame(rows,columns=['date','s','f','r']).dropna()
 ics=[]; turnovers=[]; ninst=[]
 for dt,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1:
   ics.append(spearmanr(g.f,g.r).statistic); ninst.append(len(g))
 # turnover rank changes, aligned observations
 z=x.pivot(index='date',columns='s',values='f').rank(axis=1,pct=True)
 turnovers=(z.diff().abs().mean(axis=1)*2).dropna()
 arr=np.array(ics); print(mode,'dates',len(arr),'avg_inst',np.mean(ninst),'IC',arr.mean(),'ICIR',arr.mean()/arr.std(ddof=1),'hit',np.mean(arr>0),'turn',turnovers.mean(),'coverage',len(x)/sum(len(d) for d in data.values()))
 for h in [5,10]:
  rr=[]
  for s,d in data.items():
   intr=d.close/d.open-1; f=(-intr if mode=='raw' else -intr/(intr.rolling(20).std()+1e-8)); fr=d.close.shift(-h)/d.close-1
   for dt in d.index: rr.append((dt,s,f.loc[dt],fr.loc[dt]))
  q=pd.DataFrame(rr,columns=['date','s','f','r']).dropna(); a=[]
  for dt,g in q.groupby('date'):
   if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1:a.append(spearmanr(g.f,g.r).statistic)
  a=np.array(a); print(' h',h,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1))
