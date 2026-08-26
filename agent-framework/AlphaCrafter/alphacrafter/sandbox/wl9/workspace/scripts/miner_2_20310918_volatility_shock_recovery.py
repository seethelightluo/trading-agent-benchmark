import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for s in U:
    d=None
    for fn in (get_index_daily_data, get_stock_daily_data):
        try: d=fn(s, days=5000)
        except Exception: pass
        if d is not None: break
    if d is not None and len(d):
        d=d[['date','close']].copy(); d.date=pd.to_datetime(d.date); d['lr']=np.log(d.close).diff(); raw[s]=d
px=pd.concat([v.set_index('date')['lr'].rename(k) for k,v in raw.items()],axis=1).sort_index()
cs=px.mean(axis=1)
res=px.sub(cs,axis=0)
# shock recovery: contrarian residual 20d return, emphasized when short vol is elevated vs long vol
cum=res.rolling(20,min_periods=15).sum()
short=res.rolling(5,min_periods=4).std(); long=res.rolling(60,min_periods=40).std()
shock=(short/(long+1e-8)-1).clip(-1,3)
factor=(-cum*(1+shock.clip(lower=0))).shift(1)
fwd=px.rolling(20).sum().shift(-20)
rows=[]; dates=sorted(set(factor.index)&set(fwd.index))
for dt in dates:
    a=factor.loc[dt]; b=fwd.loc[dt]; z=pd.concat([a.rename('f'),b.rename('r')],axis=1).dropna()
    if len(z)>=8:
        rows.append((dt,len(z),z.f.corr(z.r)))
out=pd.DataFrame(rows,columns=['date','n','ic']).dropna()
print('dates',len(out),'avg_n',out.n.mean(),'min_n',out.n.min(),'coverage',out.n.mean()/15)
print('IC',out.ic.mean(),'ICIR',out.ic.mean()/out.ic.std(ddof=1),'hit',(out.ic>0).mean())
for h in [5,10,20,40,60]:
    fw=px.rolling(h).sum().shift(-h); rr=[]
    for dt in factor.index:
      z=pd.concat([factor.loc[dt].rename('f'),fw.loc[dt].rename('r')],axis=1).dropna()
      if len(z)>=8: rr.append(z.f.corr(z.r))
    x=pd.Series(rr).dropna(); print('h',h,'dates',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1))
# regime
for name,lo,hi in [('2024-26','2024-01-01','2026-12-31'),('2027-29','2027-01-01','2029-12-31'),('2030','2030-01-01','2030-12-31'),('2031YTD','2031-01-01','2031-09-03')]:
 x=out[(out.date>=lo)&(out.date<=hi)].ic; print(name,len(x),x.mean(),x.mean()/x.std(ddof=1) if len(x)>1 else np.nan)
out.assign(**{s:factor.loc[out.date,s].values for s in U}).to_csv('scripts/miner_2_20310918_volatility_shock_recovery_signal.csv',index=False)
