import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:get_stock_daily_data(s,days=5000).set_index('date')['close'].astype(float) for s in syms}
P=pd.DataFrame(px).sort_index().ffill(); vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
q=vix.rolling(252,min_periods=120).rank(pct=True).shift(1)
mom=P.pct_change(20).shift(1); rel=mom.sub(mom.median(axis=1),axis=0)
F=rel.copy(); F.loc[q>=.65,:]=-rel.loc[q>=.65,:]
for h in [10,20,40,60]:
 fr=P.shift(-h)/P-1; ics=[]; ns=[]; ds=[]
 for d in F.index:
  z=pd.concat([F.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c): ics.append(c); ns.append(len(z)); ds.append(d)
 a=pd.Series(ics)
 if len(a): print(f'h={h} dates={len(a)} avg_n={np.mean(ns):.3f} coverage={np.mean(ns)/len(syms):.4f} IC={a.mean():.8f} ICIR={a.mean()/a.std():.5f} hit={(a>0).mean():.4f} start={min(ds).date()} end={max(ds).date()}')
 else: print(f'h={h} dates=0')
F.to_csv('scripts/miner_3_20350802_vix_conditional_crosssectional_reversal_signal.csv',index_label='date')
