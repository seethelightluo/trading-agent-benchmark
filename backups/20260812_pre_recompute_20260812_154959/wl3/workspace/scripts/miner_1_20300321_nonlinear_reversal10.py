import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<100: d=get_index_daily_data(s,4000)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); D[s]=d.set_index('date')
px=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index()
lr=np.log(px).diff(); rv=lr.rolling(60,min_periods=30).std()
# Nonlinear 10-session shock reversal: oppose 10d move, amplify large shocks, risk-normalized; lagged.
r10=lr.rolling(10,min_periods=10).sum()
base=(-r10/(rv*np.sqrt(10))).clip(-5,5)
sig=(base*(1+abs(base))).shift(1)
nxt=lr.shift(-1); rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt].rename('a'),nxt.loc[dt].rename('b')],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.a.corr(z.b,method='spearman'),len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('assets',len(D),'dates',len(r),'avg_n',r.n.mean(),'coverage',r.n.sum()/(len(r)*len(U)))
print('IC %.8f ICIR %.8f hit %.4f'%(r.ic.mean(),r.ic.mean()/r.ic.std(ddof=1),(r.ic>0).mean()))
for lab,sub in [('2020-22',r.loc['2020':'2022']),('2023-25',r.loc['2023':'2025']),('2026-27',r.loc['2026':'2027']),('2028-30',r.loc['2028':'2030']),('recent250',r.tail(250))]:
 if len(sub)>1: print(lab,len(sub),sub.ic.mean(),sub.ic.mean()/sub.ic.std(ddof=1),sub.n.mean())
# approximate rank turnover
ss=sig.rank(axis=1,pct=True); turn=(ss.diff().abs().mean(axis=1)).dropna(); print('rank_turnover',turn.mean())
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20300321_nonlinear_reversal10_signal.csv',index=False)
