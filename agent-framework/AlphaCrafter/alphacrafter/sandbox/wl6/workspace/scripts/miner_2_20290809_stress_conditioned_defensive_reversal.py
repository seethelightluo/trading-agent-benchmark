import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Stress-conditioned: in high lagged VIX, favor assets with low recent relative return but resilient downside profile;
# otherwise suppress signal. All inputs lagged by one day.
px={s:get_stock_daily_data(s,4000) for s in U}
vix=get_index_daily_data('VIX',4000)
cl=pd.DataFrame({s:df.set_index('date')['close'] for s,df in px.items() if df is not None}).sort_index().ffill()
vr=vix.set_index('date')['close'].sort_index().ffill()
r=cl.pct_change()
# align business observations, factor at t uses through t
common=cl.index.intersection(vr.index); cl=cl.loc[common]; r=r.loc[common]; vr=vr.loc[common]
rows=[]
for i in range(65,len(cl)-10):
    # lag VIX and stress percentile; use absolute VIX threshold + rolling percentile
    stress=float((vr.iloc[i-1]>25) or (vr.iloc[i-1]>vr.iloc[max(0,i-253):i].quantile(.75)))
    if not stress: continue
    rr=cl.iloc[i-1]/cl.iloc[i-11]-1
    med=rr.median(); rel=rr-med
    down=r.iloc[max(0,i-31):i].where(r.iloc[max(0,i-31):i]<0).std().replace(0,np.nan)
    # reversal favored, penalize unstable downside; positive sign means expected future return
    f=(-rel/(down+0.01)).replace([np.inf,-np.inf],np.nan)
    fut=cl.iloc[i+10]/cl.iloc[i]-1
    z=pd.DataFrame({'f':f,'y':fut}).dropna()
    if len(z)>=8:
        rows.append((cl.index[i],len(z),z.f.corr(z.y)))
df=pd.DataFrame(rows,columns=['date','n','ic']).dropna()
print('dates',len(df),'avg_n',df.n.mean(),'coverage',df.n.mean()/15,'IC',df.ic.mean(),'ICIR',df.ic.mean()/df.ic.std(ddof=1),'hit',(df.ic>0).mean())
for a,b in [('2020-01-01','2023-12-31'),('2024-01-01','2025-12-31'),('2026-01-01','2027-12-31'),('2028-01-01','2028-12-31'),('2029-01-01','2029-08-08')]:
 x=df[(df.date>=a)&(df.date<=b)].ic
 print(a,b,'n',len(x),'ic',x.mean() if len(x) else np.nan,'icir',x.mean()/x.std(ddof=1) if len(x)>1 else np.nan)
