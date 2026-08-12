import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def gd(s):
 d=get_stock_daily_data(s,days=4000)
 return d if d is not None else get_index_daily_data(s,days=4000)
fs={}
for s in U:
 d=gd(s)
 if d is not None and len(d): fs[s]=d.copy().assign(date=lambda z:pd.to_datetime(z.date).dt.strftime('%Y-%m-%d')).set_index('date')
cl=pd.DataFrame({s:d.close for s,d in fs.items()}).sort_index().astype(float)
vol=pd.DataFrame({s:d.volume for s,d in fs.items()}).reindex(cl.index).astype(float)
r=cl.pct_change(); r5=cl/cl.shift(5)-1
vshock=vol/(vol.rolling(20).median()+1e-12)
# reversal is stronger after unusually high recent volume; bounded log volume avoids domination
f=-r5*np.log1p(vshock.rolling(5).mean())/(r.rolling(20).std()+1e-8)
ics={h:[] for h in [1,5,10]}; ns=[]; rows=[]
for i,date in enumerate(cl.index):
 if i+10>=len(cl):continue
 x=f.loc[date].replace([np.inf,-np.inf],np.nan); ns.append(x.notna().sum())
 for h in ics:
  y=cl.shift(-h).loc[date]/cl.loc[date]-1; z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8:ics[h].append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 for s,v in x.items():rows.append({'date':date,'symbol':s,'signal':v})
print('assets',len(fs),'dates',len(ics[1]),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15)
for h,a in ics.items():
 a=np.array(a);print('H',h,'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0),'n',len(a))
pd.DataFrame(rows).to_csv('scripts/miner_3_20300418_volume_shock_reversal_signal.csv',index=False)
