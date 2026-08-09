import pandas as pd,numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-12-17')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().ffill(); P=P[P.index<=END]; R=P.pct_change()
# lagged medium-term trend gated by market breadth: asset 20d return, multiplied by fraction of other assets with positive lagged 20d return
m=R.rolling(20,min_periods=20).sum().shift(1); breadth=(m>0).mean(axis=1); F=m.mul((0.5+abs(breadth-0.5)*2),axis=0); F=F.sub(F.median(axis=1),axis=0)
F.to_csv('scripts/miner_3_20261217_breadth_trend_signal.csv',index_label='date')
for h in [1,5,10]:
 Y=P.pct_change(h).shift(-h); a=[]; ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),Y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q): a.append(q); ns.append(len(z))
 q=pd.Series(a); print('H',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),8),'ICIR',round(q.mean()/q.std(ddof=1),8),'hit',round((q>0).mean(),4))
print('coverage',F.notna().sum().sum()/F.size,'turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
cs=[]
for p in Path('scripts').glob('*signal.csv'):
 try:
  x=pd.read_csv(p,index_col=0,parse_dates=True).reindex(F.index).reindex(columns=U); c=F.stack().corr(x.stack())
  if pd.notna(c): cs.append((abs(c),p.name,c))
 except Exception: pass
print('maxcorr',max(cs) if cs else None); print('period',F.index.min().date(),F.index.max().date())
