import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
    x=get_stock_daily_data(s, days=3000)
    if x is None or len(x)==0: x=get_index_daily_data(s, days=3000)
    return x
raw={s:load(s) for s in U}
px=pd.DataFrame({s:(d.set_index('date')['close'] if d is not None else pd.Series(dtype=float)) for s,d in raw.items()}).sort_index().ffill()
# risk-adjusted medium trend: 60d return divided by 20d realized volatility, lagged naturally in factor
ret=px.pct_change()
fac=px.pct_change(60)/(ret.rolling(20).std()*np.sqrt(20))
fwd=px.shift(-10)/px-1
rows=[]
for dt in fac.index:
    a=fac.loc[dt]; b=fwd.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
    if len(z)>=8:
        rows.append((dt,len(z),z.iloc[:,0].rank().corr(z.iloc[:,1].rank())))
r=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('factor=r60/(sd20*sqrt20) dates',len(r),'avg_n',r.n.mean(),'coverage',r.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(r.ic.mean(),r.ic.mean()/r.ic.std(),(r.ic>0).mean(),fac.rank(axis=1).diff().abs().stack().mean()/14))
for name,x in [('recent180',r.tail(180)),('recent360',r.tail(360)),('2030',r.loc['2030']),('recent60',r.tail(60))]: print(name,len(x), '%.6f %.6f'%(x.ic.mean(),x.ic.mean()/x.ic.std()))
# save artifact with signal values for audit
fac.loc[r.index].to_csv('scripts/miner_3_20310123_risk_adjusted_trend_signal.csv')
r.to_csv('scripts/miner_3_20310123_risk_adjusted_trend_ic.csv')
