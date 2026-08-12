import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<100: d=get_index_daily_data(s,4000)
 if d is not None and len(d):
  x=d.copy(); x.date=pd.to_datetime(x.date); x=x.set_index('date'); frames[s]=x
px=pd.DataFrame({s:d.close for s,d in frames.items()}).sort_index()
r=np.log(px).diff()
# market stress: negative cross-sectional median and elevated cross-sectional dispersion
med=r.median(axis=1); disp=r.std(axis=1)
stress=(med < med.rolling(60,min_periods=30).quantile(.35)) & (disp > disp.rolling(60,min_periods=30).median())
# lagged 3d reversal, volatility scaled, activated only in stress
raw=-r.rolling(3).sum()/r.rolling(20).std()
sig=raw.where(stress, np.nan)
rows=[]
for dt in sig.index[:-1]:
 a=sig.loc[dt]; f=r.loc[dt:].iloc[1] if dt in r.index else None
 z=pd.concat([a,f],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,float(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')),len(z),int(stress.loc[dt])))
out=pd.DataFrame(rows,columns=['date','ic','n','stress']).set_index('date')
# note sparse signal dates; calculate daily paper observations
print('dates',len(out),'avg_n',out.n.mean(),'coverage',out.n.sum()/(len(out)*len(U)) if len(out) else 0,'stress_frac',out.stress.mean())
print('IC %.8f ICIR %.8f hit %.4f'%(out.ic.mean(),out.ic.mean()/out.ic.std(ddof=1), (out.ic>0).mean()))
for label,sub in [('2020-22',out.loc['2020':'2022']),('2023-25',out.loc['2023':'2025']),('2026-27',out.loc['2026':'2027']),('2028-30',out.loc['2028':'2030']),('recent250',out.tail(250))]:
 if len(sub)>1: print(label,len(sub),sub.ic.mean(),sub.ic.mean()/sub.ic.std(ddof=1),sub.n.mean())
# artifact
out.reset_index().to_csv('scripts/miner_1_20300221_stress_reversal_signal.csv',index=False)
