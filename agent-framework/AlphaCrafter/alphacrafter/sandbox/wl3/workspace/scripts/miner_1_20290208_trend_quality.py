import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Trend quality: 20d return multiplied by fraction of positive days, scaled by 30d volatility.
xs={}
for s in U:
    d=get_stock_daily_data(s, days=4000)
    if d is not None and len(d):
        d=d[['date','close']].copy(); d['r']=d.close.pct_change(); xs[s]=d
p=pd.concat({s:x.set_index('date').close for s,x in xs.items()},axis=1).sort_index()
r=p.pct_change()
# signal available at t, predict next 10 trading-day return
ret20=p.pct_change(20)
pos20=r.rolling(20).mean().div(r.rolling(20).std()) # risk adjusted consistency
factor=(ret20 * (r.gt(0).rolling(20).mean()-0.5) * 2).where(pos20.notna())
# standardize cross section (median demean)
factor=factor.sub(factor.median(axis=1),axis=0).shift(1)
fwd=p.shift(-10).div(p)-1
rows=[]
for dt in factor.index:
    a=factor.loc[dt]; b=fwd.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
    if len(z)>=8:
        rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for name, q in [('all',x),('2020-22',x.loc['2020':'2022']),('2023-25',x.loc['2023':'2025']),('2026-27',x.loc['2026':'2027']),('2028+',x.loc['2028':]),('recent252',x.tail(252))]:
    ic=q.ic.dropna(); print(name,'dates',len(ic),'avg_n',round(q.n.mean(),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
print('coverage',factor.notna().sum().sum()/(len(factor)*15))
# rank turnover proxy
rr=factor.rank(axis=1,pct=True); print('turnover',rr.diff().abs().mean().mean())
factor.to_csv('scripts/miner_1_20290208_trend_quality_signal.csv')
