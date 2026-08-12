import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
D={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(f)[:-4]; D[s]=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index()
C=pd.DataFrame({s:d.close for s,d in D.items()}); R=C.pct_change(); mom=C.pct_change(5)
disp=R.std(axis=1).rolling(20).mean(); breadth=(R.rolling(20).mean()>0).mean(axis=1)
scale=(disp/disp.rolling(120).median()).clip(.5,2)*(0.75+0.5*breadth)
sig=(-mom.mul(scale,axis=0)).shift(1)
res={h:[] for h in [1,5,10,20]}; ds={h:[] for h in res}
for dt in sig.index:
 for h in res:
  f=sig.loc[dt]; fr=C.shift(-h).loc[dt]/C.loc[dt]-1
  z=pd.concat([f,fr],axis=1).dropna()
  if len(z)>=8: res[h].append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ds[h].append(dt)
print('dates',len(sig),'assets',len(C.columns),'valid', {h:len(x) for h,x in res.items()})
for h,x in res.items():
 a=np.array(x); print(h, 'IC %.6f ICIR %.6f hit %.4f'%(np.mean(a),np.mean(a)/(np.std(a,ddof=1)+1e-12),np.mean(a>0)))
for lo,hi in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2031')]:
 a=np.array(res[10]); t=pd.to_datetime(ds[10]); m=(t.year>=int(lo))&(t.year<=int(hi)); q=a[m]; print(lo,hi,len(q),np.mean(q) if len(q) else np.nan,np.mean(q)/(np.std(q,ddof=1)+1e-12) if len(q)>1 else np.nan)
r=sig.rank(axis=1,pct=True); print('coverage',sig.notna().sum().sum()/sig.size,'turnover',np.nanmean((r-r.shift()).abs().mean(axis=1)))
