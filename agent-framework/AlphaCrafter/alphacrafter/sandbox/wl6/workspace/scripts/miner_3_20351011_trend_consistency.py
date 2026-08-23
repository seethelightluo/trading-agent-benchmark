import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 x=get_stock_daily_data(s,5000)
 if x is None or len(x)==0:x=get_index_daily_data(s,5000)
 return None if x is None or len(x)==0 else x[['date','close']].drop_duplicates('date').set_index('date')['close']
p={s:load(s) for s in U};p={s:x for s,x in p.items() if x is not None};C=pd.DataFrame(p).sort_index().ffill();C=C.loc[C.index<=pd.Timestamp('2035-10-10')]
r=C.pct_change(); r20=r.rolling(20).mean(); r60=r.rolling(60).mean(); v60=r.rolling(60).std()
# risk-adjusted persistent trend: blend 20D and 60D returns, penalize noisy paths.
sig=(.5*r20+.5*r60)/(v60+1e-8)
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20351011_trend_consistency_signal.csv',index=False)
for h in [5,10,20,40]:
 fw=C.shift(-h)/C-1; vals=[];ns=[]
 for d in sig.index:
  ok=sig.loc[d].notna()&fw.loc[d].notna()
  if ok.sum()>=8:
   q=sig.loc[d,ok].corr(fw.loc[d,ok],method='spearman')
   if pd.notna(q): vals.append(q);ns.append(ok.sum())
 a=pd.Series(vals);print(f'h={h} dates={len(a)} avg_inst={np.mean(ns):.3f} IC={a.mean():.8f} ICIR={a.mean()/a.std(ddof=1)*np.sqrt(len(a)):.8f} hit={(a>0).mean():.4f}')
print(f'coverage={sig.notna().sum().sum()/(len(sig)*len(U)):.6f} turnover={sig.rank(axis=1,pct=True).diff().abs().mean().mean():.6f} instruments={len(U)} dates={len(sig)} end={C.index.max().date()}')
