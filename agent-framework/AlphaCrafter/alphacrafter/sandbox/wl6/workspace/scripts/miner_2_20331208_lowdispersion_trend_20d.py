import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s, days=6000) for s in U}
prices=pd.concat({s:d.set_index('date')['close'] for s,d in D.items() if d is not None},axis=1).sort_index().ffill()
ret=prices.pct_change(); r20=prices.pct_change(20); vol20=ret.rolling(20).std()*np.sqrt(252)
disp=ret.std(axis=1); gate=disp < disp.rolling(60).median(); factor=(r20/vol20).where(gate,0.0)
for h in [5,10,20,40]:
 fwd=prices.shift(-h)/prices-1; vals=[]; ns=[]
 for dt in factor.index:
  x=factor.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8: vals.append(x[ok].corr(y[ok],method='spearman')); ns.append(ok.sum())
 a=pd.Series(vals).dropna(); ic=a.mean(); sd=a.std(ddof=1)
 print(f'h={h} dates={len(a)} avg_n={np.mean(ns):.3f} coverage={np.mean(ns)/len(U):.4f} IC={ic:.8f} ICIR={ic/sd*np.sqrt(len(a)):.8f} hit={np.mean(a>0):.4f}')
h=10; fwd=prices.shift(-h)/prices-1; vals=[]
for dt in factor.index:
 x=factor.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8: vals.append((dt,x[ok].corr(y[ok],method='spearman')))
a=pd.Series(dict(vals)); print('years10',a.groupby(a.index.year).mean().round(6).to_dict()); print('assets',len(prices.columns),'dates',len(prices),'last',prices.index[-1])
