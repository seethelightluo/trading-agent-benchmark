import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)==0: d=get_index_daily_data(s,days=3000)
 return d
raw={s:load(s) for s in U}; px=pd.DataFrame({s:(d.set_index('date').close if d is not None else pd.Series(dtype=float)) for s,d in raw.items()}).sort_index().ffill()
r=px.pct_change(); down=r.where(r<0)
# Downside-adjusted medium-term trend; min_periods avoids rejecting assets with fewer than 60 down days.
downvol=down.rolling(60,min_periods=15).std(); fac=r.rolling(60,min_periods=60).sum()/(downvol*np.sqrt(60)+1e-12); fwd=px.shift(-10)/px-1
rows=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),z.iloc[:,0].rank().corr(z.iloc[:,1].rank())))
ic=pd.DataFrame(rows,columns=['date','n','ic']); ic.date=pd.to_datetime(ic.date); ic=ic.set_index('date'); years=ic.index.year
print('dates',len(ic),'start',ic.index.min(),'end',ic.index.max(),'avg_n',ic.n.mean(),'coverage',ic.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(ic.ic.mean(),ic.ic.mean()/ic.ic.std(),(ic.ic>0).mean(),fac.rank(axis=1).diff().abs().stack().mean()/14))
for name,x in [('recent180',ic.tail(180)),('recent360',ic.tail(360)),('2030',ic[years==2030]),('2031',ic[years==2031]),('recent60',ic.tail(60))]: print(name,len(x),'IC %.6f ICIR %.6f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean()))
fac.to_csv('scripts/miner_2_20310220_downside_trend_signal.csv');ic.to_csv('scripts/miner_2_20310220_downside_trend_ic.csv')
