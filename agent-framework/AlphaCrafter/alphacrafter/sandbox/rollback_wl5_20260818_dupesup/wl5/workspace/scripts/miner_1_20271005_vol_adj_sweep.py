import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 try:D[s]=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date')
 except:pass
for h,w in [(3,10),(3,20),(5,10),(5,30),(10,20),(10,30),(7,20)]:
 rec=[]
 for s,x in D.items():
  r=x.close.pct_change(); f=-x.close.pct_change(h)/(r.rolling(w,min_periods=max(8,w//2)).std()*np.sqrt(h)+1e-12)
  for i in range(len(x)-1):
   if pd.notna(f.iloc[i]) and i+1<len(x):rec.append((x.index[i],s,f.iloc[i],x.close.iloc[i+1]/x.close.iloc[i]-1))
 a=pd.DataFrame(rec,columns=['d','s','f','y']); z=[]
 for _,g in a.groupby('d'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:z.append(spearmanr(g.f,g.y).statistic)
 z=np.array(z); print('h,w',h,w,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0))
