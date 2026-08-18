import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
CUTOFF='2028-07-13'
files=glob.glob('../persistent/stock_data/*.csv')
data={}
for f in files:
 s=f.rsplit('/',1)[-1][:-4]
 if s in ['DXY','USDCNY','USDJPY','EURUSD','VIX']: continue
 d=pd.read_csv(f); d.date=pd.to_datetime(d.date); d=d[d.date<=CUTOFF].sort_values('date')
 d['r20']=d.close.pct_change(20)
 # volume confirmation: signed trend times log relative volume, clipped for robustness
 med=d.volume.rolling(60,min_periods=30).median()
 d['factor']=d.r20*np.log((d.volume/med).clip(0.25,4.0))
 data[s]=d.set_index('date')[['close','factor']]
# common dates and cross-sectional rank IC against forward returns
all_dates=sorted(set.intersection(*[set(x.index) for x in data.values()]))
rows=[]
for dt in all_dates:
 vals=[]; rets={}
 for s,d in data.items():
  if dt not in d.index: continue
  # next completed observations (strictly after dt)
  ix=d.index.get_loc(dt)
  if ix+1>=len(d): continue
  v=d.iloc[ix].factor
  if pd.notna(v): vals.append((s,float(v)))
  rets[s]=float(d.iloc[ix+1].close/d.iloc[ix].close-1) if ix+1<len(d) else np.nan
 if len(vals)>=8:
  a=np.array([v for s,v in vals]); b=np.array([rets[s] for s,v in vals])
  if np.isfinite(b).sum()>=8 and np.std(a)>0 and np.std(b)>0:
   rows.append((dt,spearmanr(a,b,nan_policy='omit').statistic,len(vals)))
x=pd.DataFrame(rows,columns=['date','ic','n']).dropna()
print('factor=20d return * clipped log(volume/60d median); cutoff',CUTOFF)
print('dates',len(x),'instruments',len(data),'mean_n',x.n.mean(),'coverage',x.n.mean()/len(data))
print('1d IC %.6f ICIR %.6f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean()))
for h in [5,10,20]:
 rows=[]
 for dt in all_dates:
  vals=[]; rets=[]
  for s,d in data.items():
   if dt not in d.index: continue
   ix=d.index.get_loc(dt)
   if ix+h>=len(d): continue
   v=d.iloc[ix].factor; r=d.iloc[ix+h].close/d.iloc[ix].close-1
   if pd.notna(v): vals.append(float(v)); rets.append(float(r))
  if len(vals)>=8 and np.std(vals)>0 and np.std(rets)>0: rows.append(spearmanr(vals,rets).statistic)
 z=pd.Series(rows).dropna(); print('%dd dates %d IC %.6f ICIR %.6f hit %.4f'%(h,len(z),z.mean(),z.mean()/z.std(),(z>0).mean()))
print('recent250',x.tail(250).ic.mean(),x.tail(250).ic.mean()/x.tail(250).ic.std())
