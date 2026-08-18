import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

u=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
series={}
for s in u:
    d=get_stock_daily_data(s, days=5000)
    if d is not None and len(d):
        x=d[['date','close']].copy(); x['date']=pd.to_datetime(x.date); series[s]=x.set_index('date').close
px=pd.DataFrame(series).sort_index().ffill()
# one candidate: volatility-normalized oversold rebound, activated only after broad stress
ret=px.pct_change()
bench=ret.mean(axis=1)
# all signals lagged one completed session via shift
r5=px.pct_change(5); r20=px.pct_change(20); vol=ret.rolling(20).std();
# residual short return against equal weight benchmark, and stress activation
res5=r5.sub(r5.mean(axis=1),axis=0)
stress=(bench.rolling(20).sum()<0) | (res5.std(axis=1)>res5.std(axis=1).rolling(120).quantile(.60))
factor=(-r5).shift(1)
fwd=px.shift(-10)/px-1
rows=[]
for dt in factor.index:
    a=factor.loc[dt]; b=fwd.loc[dt]
    z=pd.concat([a,b],axis=1).dropna()
    if len(z)>=8:
        rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z),stress.loc[dt]))
r=pd.DataFrame(rows,columns=['date','ic','n','active']).set_index('date')
# retain complete validation after sufficient warmup
r=r.loc['2020-01-01':'2033-07-21']
ics=r.ic.dropna(); ic=ics.mean(); sd=ics.std(ddof=1); icir=ic/sd*np.sqrt(252) if sd else np.nan
# daily ICIR gate in benchmark convention appears mean/std (prior reports use this); print both
print('dates',len(r),'avg_n',r.n.mean(),'coverage',r.n.mean()/15,'active_fraction',r.active.mean())
print('IC',ic,'ICIR_daily',ic/sd,'ICIR_annualized',icir,'hit', (ics>0).mean())
print('turnover_proxy', (factor.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for a,b in [('2020-2023',('2020-01-01','2023-12-31')),('2024-2026',('2024-01-01','2026-12-31')),('2027-2029',('2027-01-01','2029-12-31')),('2030-2032',('2030-01-01','2032-12-31')),('2033',('2033-01-01','2033-07-21'))]:
    q=r.loc[b[0]:b[1]]; print(a,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1) if len(q)>2 else np.nan)
for h in [5,10,20]:
    fw=px.shift(-h)/px-1; rr=[]
    for dt in factor.index:
        z=pd.concat([factor.loc[dt],fw.loc[dt]],axis=1).dropna()
        if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
    print('horizon',h,'IC',np.nanmean(rr),'n',len(rr))
factor.to_csv('scripts/miner_1_20330722_stress_oversold_rebound_signal.csv')
