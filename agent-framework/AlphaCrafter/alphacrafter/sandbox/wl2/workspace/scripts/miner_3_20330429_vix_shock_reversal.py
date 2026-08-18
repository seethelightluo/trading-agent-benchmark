import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv');d.date=pd.to_datetime(d.date);return d.set_index('date').close.sort_index()
px=pd.concat({s:load(s) for s in U},axis=1); r=np.log(px).diff()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(r.index)
vr=vix.pct_change(); q=vr.rolling(120,min_periods=60).quantile(.8)
# reversal only after a lagged sharp VIX rise, with cross-sectional common shock removed
cs=r.rolling(3).sum(); med=cs.median(axis=1); vol=r.rolling(20,min_periods=15).std()
f=(-cs.sub(med,axis=0).div(vol*np.sqrt(3))).where(vr>q).shift(1)
rows=[]
for d in r.index:
 z=pd.concat([f.loc[d],r.shift(-1).loc[d]],axis=1).dropna()
 if len(z)>=8:rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(x),'assets',15,'coverage',x.n.mean()/15,'active',(f.notna().any(axis=1)).sum(),'IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(ddof=1),'hit',(x.ic>0).mean())
for a,b in [('2026','2029-12-31'),('2030','2033-04-29')]:
 z=x.loc[a:b];print(a,len(z),z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1) if len(z)>1 else np.nan)
for h in [1,3,5,10]:
 y=r.rolling(h).sum().shift(-h);a=[]
 for d in r.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,len(a),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1))
f.to_csv('scripts/miner_3_20330429_vix_shock_reversal_signal.csv')
