import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for f in (get_stock_daily_data,get_index_daily_data):
  try: x=f(s,5000)
  except (FileNotFoundError,KeyError): x=None
  if x is not None and len(x): return x[['date','close']].drop_duplicates('date').set_index('date')['close']
 return None
p={s:load(s) for s in U}; C=pd.DataFrame({s:x for s,x in p.items() if x is not None}).sort_index().ffill(); C=C.loc[C.index<=pd.Timestamp('2035-12-05')]
macro=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).drop_duplicates('date').set_index('date')['close'].reindex(C.index).ffill()
r=C.pct_change(); raw=C.pct_change(20); resid=raw.sub(raw.mean(axis=1),axis=0); risk=r.rolling(60).std(); base=resid/(risk+1e-8)
z=(macro.pct_change(20)-macro.pct_change(20).rolling(252).mean())/(macro.pct_change(20).rolling(252).std()+1e-8)
sig=base*(1+0.35*z.clip(-2,2).to_numpy()[:,None]); sig=sig.sub(sig.mean(axis=1),axis=0)
for h in [5,10,20,40]:
 fw=C.shift(-h)/C-1; vals=[]; ns=[]
 for i in range(len(sig)):
  a=sig.iloc[i].to_numpy(); b=fw.iloc[i].to_numpy(); ok=np.isfinite(a)&np.isfinite(b)
  if ok.sum()>=8:
   ra=pd.Series(a[ok]).rank().to_numpy(); rb=pd.Series(b[ok]).rank().to_numpy(); q=np.corrcoef(ra,rb)[0,1]
   if np.isfinite(q): vals.append(q); ns.append(ok.sum())
 a=np.asarray(vals); print(f'h={h} dates={len(a)} avg_inst={np.mean(ns):.3f} IC={a.mean():.8f} ICIR={a.mean()/a.std(ddof=1)*np.sqrt(len(a)):.8f} hit={(a>0).mean():.4f}')
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20351206_dxy_regime_residual_momentum_signal.csv',index=False)
print(f'coverage={sig.notna().sum().sum()/(len(sig)*len(U)):.6f} turnover={sig.rank(axis=1,pct=True).diff().abs().mean().mean():.6f} instruments={len(U)} dates={len(sig)} end={C.index.max().date()}')
