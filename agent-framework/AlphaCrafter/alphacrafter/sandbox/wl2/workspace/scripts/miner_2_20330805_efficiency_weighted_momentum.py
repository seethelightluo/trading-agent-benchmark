import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Efficiency-weighted medium trend: signed 20d return, scaled by directional efficiency and risk.
px={}
for s in U:
    d=get_stock_daily_data(s, days=4000)
    if d is None or len(d)<150: d=get_index_daily_data(s, days=4000)
    if d is not None: px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill()
r=P.pct_change()
ret20=P/P.shift(20)-1
eff20=ret20.abs()/(r.abs().rolling(20).sum())
vol20=r.rolling(20).std()
f=ret20*eff20/(vol20*np.sqrt(20))
f=f.replace([np.inf,-np.inf],np.nan)
fwd=P.shift(-10)/P-1
rows=[]
for dt in f.index:
    x=f.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
ic=pd.Series({d:v for d,v,n in rows})
print('dates',len(ic),'avg_n',np.mean([n for d,v,n in rows]),'coverage',len(ic)/len(f))
print('IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit', (ic>0).mean(),'abs gates',abs(ic.mean()),abs(ic.mean()/ic.std(ddof=1)))
for name,sel in [('2020-25',ic.index<'2026-01-01'),('2026-29',(ic.index>='2026-01-01')&(ic.index<'2030-01-01')),('2030+',ic.index>='2030-01-01'),('last365',ic.index>=ic.index.max()-pd.Timedelta(days=365))]:
 q=ic[sel]; print(name,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>2 else np.nan,(q>0).mean())
# signal artifact needed for audit
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20330805_efficiency_weighted_momentum_signal.csv',index=False)
# turnover based on cross-sectional ranks
rank=f.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna()
print('turnover',turn.mean(),'valid_assets',P.notna().mean().mean())
