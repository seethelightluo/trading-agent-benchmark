import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 x=get_stock_daily_data(s,5000)
 if x is None or len(x)==0: x=get_index_daily_data(s,5000)
 return None if x is None or len(x)==0 else x[['date','close']].drop_duplicates('date').set_index('date')['close']
p={s:load(s) for s in U}; p={s:x for s,x in p.items() if x is not None}
C=pd.DataFrame(p).sort_index().ffill(); C=C.loc[C.index<=pd.Timestamp('2035-12-19')]
r=C.pct_change(); r5=C/C.shift(5)-1; v20=r.rolling(20).std(); v120=r.rolling(120).std()
# Volatility-shock rebound: fade recent relative losers, but only when their own volatility
# has sharply jumped versus baseline; scaling by volatility limits unstable raw moves.
resid=r5.sub(r5.mean(axis=1),axis=0)
shock=(v20/(v120+1e-12)).clip(0.5,3.0)
sig=(-resid/(v20+1e-8))*shock
sig=sig.replace([np.inf,-np.inf],np.nan)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20351220_volatility_shock_rebound_signal.csv',index=False)
for h in [5,10,20,40]:
 fw=C.shift(-h)/C-1; vals=[]; ns=[]
 for d in sig.index:
  ok=sig.loc[d].notna()&fw.loc[d].notna()
  if ok.sum()>=8:
   q=sig.loc[d,ok].corr(fw.loc[d,ok],method='spearman')
   if pd.notna(q): vals.append(q); ns.append(ok.sum())
 a=pd.Series(vals); print(f'h={h} dates={len(a)} avg_inst={np.mean(ns):.3f} IC={a.mean():.8f} ICIR={a.mean()/a.std(ddof=1)*np.sqrt(len(a)):.8f} hit={(a>0).mean():.4f}')
print(f'coverage={sig.notna().sum().sum()/(len(sig)*len(U)):.6f} turnover={sig.rank(axis=1,pct=True).diff().abs().mean().mean():.6f} instruments={len(U)} dates={len(sig)} end={C.index.max().date()}')
