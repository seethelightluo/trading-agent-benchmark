import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def L(s):
 d=get_stock_daily_data(s,5000); d=d.copy(); d.date=pd.to_datetime(d.date); return d.drop_duplicates('date').set_index('date').sort_index().close.astype(float)
P=pd.DataFrame({s:L(s) for s in U}).sort_index(); R=P.pct_change(); r5=P.pct_change(5); v=R.rolling(20).std(); disp=R.rolling(5).std().mean(axis=1); gate=(disp.shift(1)>disp.shift(1).rolling(252,min_periods=100).quantile(.70)); med=r5.median(axis=1); F=(-(r5.sub(med,axis=0)/v).shift(1)).where(gate, np.nan)
res={h:[] for h in [1,3,5,10]}; active=0; cov=[]; dates=[]
for i in range(len(P)-10):
 x=F.iloc[i]
 if x.notna().sum()>=8:
  active+=1;dates.append(P.index[i]);cov.append(x.notna().mean())
  for h in res:
   z=pd.concat([x,P.iloc[i+h].div(P.iloc[i])-1],axis=1).dropna()
   if len(z)>=8:res[h].append(z.iloc[:,0].corr(z.iloc[:,1]))
print('active',active,'assets',len(U),'coverage',np.mean(cov))
for h,a in res.items():
 a=np.array(a);print(h,'n',len(a),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0))
F.index.name='date';F.to_csv('scripts/miner_3_20330106_dispersion70_reversal_signal.csv')
