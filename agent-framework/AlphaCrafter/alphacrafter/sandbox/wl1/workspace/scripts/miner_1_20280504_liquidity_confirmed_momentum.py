import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

asof='2028-05-04'
acct=get_account_dict(); syms=acct.get('watch_list',[])
if not syms: syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in syms:
    d=get_stock_daily_data(s,3000)
    if d is None or len(d)<80: d=get_index_daily_data(s,3000)
    if d is not None and len(d)>0:
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.set_index('date').sort_index()
        frames[s]=d
prices=pd.DataFrame({s:d['close'] for s,d in frames.items()})
vol=pd.DataFrame({s:d['volume'] for s,d in frames.items()})
# Liquidity-confirmed intermediate momentum: 20d return, scaled by relative recent activity.
ret=prices.pct_change(20)
activity=(vol.rolling(5,min_periods=3).mean()/(vol.rolling(20,min_periods=10).mean()+1e-12)).clip(0.25,4)
factor=(ret*activity).shift(1)
fwd=prices.pct_change(10).shift(-10)
rows=[]; dates=sorted(set(factor.index)&set(fwd.index))
for dt in dates:
    x=factor.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
        rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
r=pd.DataFrame(rows,columns=['date','n','ic']).dropna()
print('symbols',len(frames),'dates',len(r),'avgN',r.n.mean(),'coverage',len(frames)/15)
for h in [5,10,20]:
    fy=prices.pct_change(h).shift(-h); rr=[]
    for dt in factor.index:
        z=pd.concat([factor.loc[dt],fy.loc[dt]],axis=1).dropna()
        if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
    a=pd.Series(rr).dropna(); print('horizon',h,'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1), (a>0).mean()))
print('turnover',factor.rank(axis=1,pct=True).diff().abs().mean().mean())
for label,sub in [('2026+',r[r.date>='2026-01-01']),('2027+',r[r.date>='2027-01-01']),('2028YTD',r[r.date>='2028-01-01'])]:
 print(label,'dates',len(sub),'IC %.6f ICIR %.6f'%(sub.ic.mean(),sub.ic.mean()/sub.ic.std(ddof=1)))
r.to_csv('scripts/miner_1_20280504_liquidity_confirmed_momentum_signal.csv',index=False)
