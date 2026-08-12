import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    for fn in (get_index_daily_data,get_stock_daily_data):
        try:
            d=fn(s, days=3000)
            if d is not None and len(d)>=300:
                d=d.copy(); d['date']=pd.to_datetime(d['date']); return d.set_index('date').sort_index()['close'].astype(float)
        except (FileNotFoundError, Exception):
            continue
    return None
px={s:fetch(s) for s in U}; px={s:x for s,x in px.items() if x is not None}
P=pd.concat(px,axis=1).sort_index().ffill(); R=P.pct_change()
ret10=P/P.shift(10)-1; neg=R.clip(upper=0).rolling(20).std()
f=-(ret10/(neg*np.sqrt(252)+1e-8)); f=f.sub(f.median(axis=1),axis=0).clip(-6,6)
for h in [1,3,5,10]:
 fr=P.shift(-h)/P-1; vals=[]; dates=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1])); dates.append(dt); ns.append(len(z))
 ic=pd.Series(vals,index=dates).dropna()
 print('h',h,'obs',len(ic),'avgN',round(float(np.mean(ns)),2),'IC',round(float(ic.mean()),6),'ICIR',round(float(ic.mean()/ic.std(ddof=1)*np.sqrt(len(ic))),6),'hit',round(float((ic>0).mean()),4))
 if h==1:
  print('coverage',round(float(f.notna().mean().mean()),4),'rank_turnover',round(float(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()),6))
  for label,start,end in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-27','2025-01-01','2027-06-30')]:
   x=ic[(ic.index>=start)&(ic.index<=end)]; print(label,len(x),round(float(x.mean()),6),round(float(x.mean()/x.std(ddof=1)*np.sqrt(len(x))),4) if len(x)>1 else np.nan)
out=f.dropna(how='all').iloc[-1].rename('signal').to_frame(); out.index.name='symbol'; out.to_csv('scripts/miner_2_20270701_downside_asymmetry_reversal_signal.csv')
print('dates',P.index.min(),P.index.max(),'assets',len(P.columns),'cutoff',f.dropna(how='all').index.max(),'symbols',list(P.columns))
