import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
files=glob.glob('../persistent/stock_data/*.csv')
d={}
for f in files:
 s=os.path.basename(f)[:-4]; x=pd.read_csv(f); x['date']=pd.to_datetime(x.date); x=x.sort_values('date').set_index('date'); d[s]=x.close
px=pd.DataFrame(d).sort_index(); r=px.pct_change()
# acceleration: recent 5d return minus prior 15d average daily return, then rank cross-section
f=r.rolling(5).sum()-r.shift(5).rolling(15).sum()/3
# robust trend acceleration scaled by 20d vol, only information through date t
f=f/(r.rolling(20).std()*np.sqrt(20)+1e-9)
rows=[]
for h in [1,5,10]:
 ic=[]; n=[]; turnover=[]; prev=None
 # forward compounded return, aligned t -> t+h
 fr=px.shift(-h)/px-1
 for date in px.index:
  a=f.loc[date]; y=fr.loc[date]; z=pd.concat([a,y],axis=1).dropna()
  if len(z)>=8:
   ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); n.append(len(z))
   if h==1:
    ranks=a.rank(pct=True)
    if prev is not None: turnover.append(np.mean(np.abs(ranks-prev)))
    prev=ranks
 arr=np.array(ic); print(h,'dates',len(arr),'avg_n',np.mean(n),'coverage',np.mean(n)/15,'IC',np.nanmean(arr),'ICIR',np.nanmean(arr)/(np.nanstd(arr,ddof=1)+1e-12),'hit',np.mean(arr>0),'turn',np.mean(turnover) if turnover else None)
# regime blocks
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026-07-15'),('2026-07-16','2027-02-25')]:
 fr=px.shift(-1)/px-1; a=[]
 for dt in px.loc[lo:hi].index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(lo,hi,'n',len(a),'ic',np.mean(a) if a else None)
print('instruments',len(px.columns),'dates',px.index.min(),px.index.max())
# save signal artifact
out=f.copy(); out.index.name='date'; out.to_csv('../persistent/factor_signals_miner_3_20270225_trend_acceleration_v2.csv')
