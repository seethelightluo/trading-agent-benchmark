import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
A=[]
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None: continue
 d=d.copy();d.date=pd.to_datetime(d.date);d=d.drop_duplicates('date').set_index('date').sort_index()
 r=d.close.pct_change(); vs=d.volume/d.volume.rolling(20,min_periods=15).median()-1
 sig=r.rolling(5,min_periods=5).sum()*vs.clip(-2,2)
 fr=d.close.shift(-1)/d.close-1
 A.append(pd.DataFrame({'date':d.index,'sig':sig,'fr':fr}).dropna().assign(s=s))
x=pd.concat(A).reset_index(drop=True); z=[]; ns=[]
for dt,g in x.groupby('date'):
 if len(g)>=8 and g.sig.nunique()>1:z.append(g.sig.corr(g.fr,method='spearman'));ns.append(len(g))
z=np.array(z);print('dates',len(z),'avg_n',np.mean(ns),'IC',np.nanmean(z),'ICIR',np.nanmean(z)/np.nanstd(z,ddof=1),'hit',np.mean(z>0),'coverage_rows',len(x))
print('valid rows',len(x),'assets',x.s.nunique())
x.to_csv('scripts/miner_3_20261022_volume_confirm_signal.csv',index=False)
