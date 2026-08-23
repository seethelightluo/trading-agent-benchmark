import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:x=fn(s,days=3000);break
  except (FileNotFoundError,KeyError):pass
 if x is not None: D[s]=x.assign(date=pd.to_datetime(x.date)).set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff(); ret5=p.pct_change(5); disp=ret5.std(axis=1); q=disp.rolling(120).rank(pct=True)
# reversal only amplified during elevated cross-sectional dispersion, interpretable continuous signal
f=-ret5.mul((0.5+q),axis=0); fr=p.shift(-10).div(p)-1
ic=[]; ds=[]; ns=[]
for d in f.index:
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8:ic.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ds.append(d);ns.append(len(z))
ic=np.array(ic);print('dates',len(ic),'avg_n',np.mean(ns),'coverage',len(ic)/len(f));print('IC %.6f ICIR %.6f hit %.4f'%(np.mean(ic),np.mean(ic)/np.std(ic,ddof=1),np.mean(ic>0)))
for y0,y1 in [('2020','2023'),('2023','2025'),('2025','2027'),('2027','2029'),('2029','2030')]:
 a=ic[(np.array(ds)>=pd.Timestamp(y0))&(np.array(ds)<pd.Timestamp(y1))];print(y0+'-'+y1,len(a),np.mean(a) if len(a) else np.nan)
print('turnover',f.rank(pct=True).diff().abs().mean(axis=1).dropna().mean())
