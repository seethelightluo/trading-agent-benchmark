import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
cutoff=pd.Timestamp('2030-01-23'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b=Path('../persistent/stock_data')
cs={s:pd.read_csv(b/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index().loc[:cutoff] for s in syms}; ix=sorted(set().union(*[x.index for x in cs.values()])); p=pd.DataFrame({s:cs[s].reindex(ix) for s in syms}); r=p.pct_change(); r1=r.sub(r.median(axis=1),axis=0); vol20=r.rolling(20,min_periods=15).std()*np.sqrt(252)
# One-day residual reversal, volatility-normalized, with a mild medium-term trend
# confirmation gate: fade shocks only when 20d trend is not strongly opposed.
trend=p/p.shift(20)-1; gate=np.where(trend>=-0.08,1.0,0.25); sig=(-r1/(1+vol20)*gate).shift(1); sig=pd.DataFrame(sig,index=p.index,columns=p.columns); sig.to_csv('scripts/miner_2_20300124_daily_residual_reversal_signal.csv')
print('coverage %.6f turnover %.6f'%(sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
for h in [1,5,10,20]:
 f=p.shift(-h)/p-1; z=[];ns=[];ds=[]
 for d in ix:
  ok=sig.loc[d].notna()&f.loc[d].notna()
  if ok.sum()>=8:z.append(spearmanr(sig.loc[d,ok],f.loc[d,ok]).statistic);ns.append(ok.sum());ds.append(d)
 z=np.asarray(z);print(f'H {h} dates {len(z)} avgN {np.mean(ns):.2f} IC {z.mean():.6f} ICIR {z.mean()/z.std(ddof=1):.6f} hit {np.mean(z>0):.4f}')
