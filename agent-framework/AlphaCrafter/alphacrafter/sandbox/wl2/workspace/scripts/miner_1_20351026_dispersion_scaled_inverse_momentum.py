import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s,days=4000)
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); D[s]=x.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill()
r=p.pct_change()
# candidate: inverse medium momentum, amplified in high cross-sectional dispersion,
# with a smooth high-volatility regime gate (observable through t-1)
csdisp=r.rolling(20).std().mean(axis=1)
volstate=(csdisp/csdisp.rolling(120).median()).clip(0.5,2.0)
base=-(p.pct_change(60))/r.rolling(20).std()
f=base.mul((volstate.clip(0.8,1.5)**0.35),axis=0)
# one-day lag and 10-session forward return
sig=f.shift(1); fr=p.shift(-10)/p-1
rows=[]
for d in sig.index:
    a=sig.loc[d]; b=fr.loc[d]; z=pd.concat([a,b],axis=1).dropna()
    if len(z)>=8:
        rows.append((d,z.iloc[:,0].corr(z.iloc[:,1]),len(z),a.notna().mean()))
out=pd.DataFrame(rows,columns=['date','ic','n','coverage']).set_index('date')
# exclude unstable initial period only via valid naturally
ic=out.ic.dropna(); print('dates',len(out),'avg_n',out.n.mean(),'coverage',out.coverage.mean())
print('IC %.6f ICIR %.6f hit %.4f turnover %.4f'%(ic.mean(),ic.mean()/ic.std(),(ic>0).mean(),sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for a,b in [('2026-01-01','2028-12-31'),('2029-01-01','2032-12-31'),('2033-01-01','2035-10-15'),('2035-01-01','2035-10-15')]:
 q=ic.loc[a:b]; print(a,b,len(q),q.mean(),q.mean()/q.std() if len(q)>1 else np.nan)
# artifact
sig.loc[out.index].to_csv('../persistent/miner_1_20351026_dispersion_scaled_inverse_momentum_signal.csv')
out.to_csv('../persistent/miner_1_20351026_dispersion_scaled_inverse_momentum_ic.csv')
