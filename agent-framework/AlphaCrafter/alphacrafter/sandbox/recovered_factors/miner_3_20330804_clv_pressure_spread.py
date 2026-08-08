import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
from pathlib import Path
A={Path(f).stem:pd.read_csv(f,parse_dates=['date']).set_index('date') for f in glob.glob('../persistent/stock_data/*.csv')}
# Novel path-shape factor: recent close-location pressure minus slow pressure (mean-reversion of bar positioning)
s={}
for a,d in A.items():
 den=(d.high-d.low).replace(0,np.nan); clv=(2*d.close-d.high-d.low)/den
 s[a]=(clv.rolling(5,min_periods=4).mean()-clv.rolling(20,min_periods=15).mean()).shift(1)
s=pd.DataFrame(s); c=pd.DataFrame({a:d.close for a,d in A.items()}).sort_index(); s=s.reindex(c.index)
for h in [1,5,10,20]:
 f=c.shift(-h)/c-1;x=[]; nn=[]
 for dt in s.index:
  z=pd.concat([s.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);nn.append(len(z))
 x=np.array(x);print('H',h,'dates',len(x),'meanN',round(np.mean(nn),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4))
print('assets',len(A),'range',s.index.min(),s.index.max(),'coverage',s.notna().sum().sum()/(s.shape[0]*s.shape[1]))
print('turnover10',s.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean())
f=c.shift(-10)/c-1
for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2033')]:
 x=[]
 for dt in s.loc[lo:hi].index:
  z=pd.concat([s.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=np.array(x);print('regime',lo,hi,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6))
