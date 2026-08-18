import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 try:x=get_stock_daily_data(s,4000)
 except: x=None
 if x is not None and len(x):
  x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index();R=np.log(P/P.shift(1))
for look in [1,3,5,10]:
 for h in [1,5,10]:
  a=[]
  for i in range(25,len(P)-h):
   f=(-R.iloc[i-look+1:i+1].sum()).div(R.iloc[i-19:i+1].std()+1e-8)
   y=(P.shift(-h).iloc[i]/P.iloc[i]-1); z=pd.concat([f,y],axis=1).dropna()
   if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
  a=np.array(a);print('look',look,'h',h,'n',len(a),'IC',round(a.mean(),5),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a)),5),'hit',round((a>0).mean(),3),'recent',round(a[-250:].mean(),5))
print('dates',P.index.min(),P.index.max(),'instruments',len(P.columns),'avg',round(P.notna().sum(axis=1).mean(),2))
