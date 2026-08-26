import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
    x=get_stock_daily_data(a, days=5000)
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); D[a]=x.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill()
r=np.log(p).diff()
# Causal relative downside-beta reversal: residual 30d return against cross-asset median,
# scaled by asset-specific downside deviation and amplified only during broad stress.
med=r.median(axis=1)
res=r.sub(med,axis=0)
res30=res.rolling(30,min_periods=25).sum()
down=res.where(res<0).rolling(60,min_periods=45).std()
market=-med.rolling(20,min_periods=15).sum()/(med.rolling(60,min_periods=45).std()*np.sqrt(20))
stress=market.clip(0,2)
f=(-res30/(down*np.sqrt(30))).mul(1+0.35*stress, axis=0)
f=f.replace([np.inf,-np.inf],np.nan)
fwd=np.log(p.shift(-10)/p)
ics=[]; turns=[]; ninst=[]; dates=[]
for d in f.index:
    z=f.loc[d]; y=fwd.loc[d]; ok=z.notna()&y.notna()
    if ok.sum()>=8:
        ics.append(z[ok].corr(y[ok])); ninst.append(ok.sum()); dates.append(d)
        turns.append((z[ok].rank(pct=True)-f.shift(1).loc[d][ok].rank(pct=True)).abs().mean())
ics=pd.Series(ics,index=pd.to_datetime(dates)).dropna()
print('dates',len(ics),'mean_inst',np.mean(ninst),'coverage',np.mean(ninst)/15,'IC',ics.mean(),'ICIR',ics.mean()/ics.std(),'hit',np.mean(ics>0),'turnover',np.nanmean(turns))
for label,lo,hi in [('2020-24','2020','2025'),('2025-27','2025','2028'),('2028-29','2028','2030'),('2030-33','2030','2034')]:
 q=ics[(ics.index>=lo)&(ics.index<hi)]; print(label,len(q),q.mean(),q.mean()/q.std() if len(q)>1 else np.nan)
# signal artifact
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20330609_relative_downside_beta_reversal_signal.csv',index=False)
