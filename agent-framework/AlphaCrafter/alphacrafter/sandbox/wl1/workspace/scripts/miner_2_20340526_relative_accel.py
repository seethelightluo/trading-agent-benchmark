import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try:
        x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close
        D[s]=x.astype(float)
    except Exception as e: print('missing',s,e)
p=pd.DataFrame(D).sort_index().ffill().loc[:'2034-05-26']
r=p.pct_change()
# market-relative medium-term acceleration: residual 40d return versus cross-sectional equal-weight market,
# minus residual 120d/3; lagged one day
m=r.mean(axis=1)
rm=m.rolling(40).sum(); rM=m.rolling(120).sum()
sig=r.rolling(40).sum().sub(rm,axis=0).sub((r.rolling(120).sum().sub(rM,axis=0))/3,axis=0).shift(1)
# forward compounded returns by exact calendar rows
metrics=[]; dec={}
for h in [5,10,20,40]:
    fwd=p.shift(-h)/p-1
    vals=[]
    for dt in sig.index:
        a=sig.loc[dt]; b=fwd.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
        if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
    q=pd.Series(vals).dropna(); dec[h]=(q.mean(),q.mean()/q.std(ddof=1), (q>0).mean(),len(q))
q=pd.Series([x[0] for x in []])
# daily paper IC is h=1; admission same-horizon requested, use 10d IC and daily ICIR convention from daily IC observations
fwd=p.shift(-10)/p-1; daily=[]; cov=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: daily.append(z.iloc[:,0].corr(z.iloc[:,1])); cov.append(len(z)/15)
ic=pd.Series(daily).dropna();
# report both daily IC and 10d same horizon IC
print('dates',len(ic),'avgN',np.mean(np.array(cov)*15),'coverage',np.mean(cov),'daily_paper_IC',ic.mean(),'daily_ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean(),'turnover',sig.diff().rank(pct=True).diff().abs().mean().mean())
print('decay',dec)
for start,end in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 q=ic[(ic.index.astype(str)>=start)&(ic.index.astype(str)<=end)]
 print(start,end,len(q),q.mean() if len(q) else np.nan,(q.mean()/q.std(ddof=1)) if len(q)>2 else np.nan)
# artifact
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20340526_relative_accel_signal.csv',index=False)
