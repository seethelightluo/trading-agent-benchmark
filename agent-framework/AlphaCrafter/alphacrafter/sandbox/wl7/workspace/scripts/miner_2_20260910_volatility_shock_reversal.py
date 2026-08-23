import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data, get_account_dict

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load(s):
    d=get_stock_daily_data(s,2500)
    if d is None or len(d)==0: d=get_index_daily_data(s,2500)
    x=d[['date','close']].copy(); x['date']=pd.to_datetime(x.date); return x.set_index('date').close
p=pd.concat({s:load(s) for s in U},axis=1).sort_index().ffill()
r=p.pct_change()
# Candidate: volatility-shock reversal. A large one-day move relative to its own
# recent volatility tends to mean-revert across this heterogeneous universe.
vol=r.rolling(20,min_periods=15).std()
f=(-r.shift(1)/vol.shift(1)).replace([np.inf,-np.inf],np.nan)
# evaluate forward returns, each date cross-section, with >=8 names
out=[]
for h in [1,5,10,20]:
    fr=p.shift(-h)/p-1
    vals=[]; dates=[]; nms=[]
    for dt in f.index:
        a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
        if len(a)>=8:
            vals.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman')); dates.append(dt); nms.append(len(a))
    ic=pd.Series(vals,index=dates).dropna()
    print('H',h,'dates',len(ic),'avg_n',round(np.mean(nms),2),'IC',round(ic.mean(),5),'ICIR',round(ic.mean()/ic.std(ddof=1),5),'hit',round((ic>0).mean(),4))
    if h==1:
      sig=f.rank(axis=1,pct=True); turnover=sig.diff().abs().mean(axis=1).mean(); cov=f.notna().sum(axis=1).mean()/15
      print('coverage',round(cov,4),'turnover',round(turnover,4),'period',ic.index.min().date(),ic.index.max().date())
      for name,(a,b) in {'2020-22':('2020','2022-12-31'),'2023-24':('2023','2024-12-31'),'2025-26':('2025','2026-12-31')}.items():
        z=ic.loc[a:b]; print('REG',name,'n',len(z),'ICIR',round(z.mean()/z.std(ddof=1),4) if len(z)>1 else None,'IC',round(z.mean(),5) if len(z) else None)
print('candidate=volatility_shock_reversal')
