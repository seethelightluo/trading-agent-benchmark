import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x['date']=pd.to_datetime(x.date); D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# path-efficiency trend: net 30d return divided by total absolute daily path, lagged one day
f=(p.pct_change(30)/r.abs().rolling(30,min_periods=20).sum()).shift(1)
for h in [1,5,10,20]:
 out=[]
 for i in range(45,len(p)-h):
  z=pd.concat([f.iloc[i],p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(out,index=p.index[45:45+len(out)])
 print('h',h,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252/h),'hit',(q>0).mean())
# use exact 10d regime slices by dates through current available
print('coverage',f.notna().sum().mean()/15,'turnover',f.rank(pct=True).diff().abs().mean().mean())
