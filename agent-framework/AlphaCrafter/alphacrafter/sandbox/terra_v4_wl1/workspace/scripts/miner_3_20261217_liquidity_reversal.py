import pandas as pd,numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().ffill(); P=P[P.index<=cut]; R=P.pct_change(); V=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['volume'] for s in U}).reindex(P.index).ffill()
# liquidity-confirmed reversal: fade 3d move, amplified when volume is below its lagged 20d median (overreaction less confirmed)
ret3=R.rolling(3,min_periods=3).sum(); vr=V/V.rolling(20,min_periods=15).median(); F=-ret3*(2-vr.clip(0,2)); F=F.sub(F.median(axis=1),axis=0)
F.to_csv('scripts/miner_3_20261217_liquidity_reversal_signal.csv',index_label='date')
for h in [1,5,10]:
 Y=P.pct_change(h).shift(-h); vals=[];ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),Y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: vals.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
 a=pd.Series(vals); print('H',h,'dates',len(a),'avg_names',round(np.mean(ns),2),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1),8),'hit',round((a>0).mean(),4))
print('coverage',round(F.notna().sum().sum()/F.size,6),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
cs=[]
for p in Path('scripts').glob('*signal.csv'):
 try:
  x=pd.read_csv(p,index_col=0,parse_dates=True).reindex(F.index).reindex(columns=U);c=F.stack().corr(x.stack())
  if pd.notna(c):cs.append((abs(c),p.name,c))
 except:pass
print('max_abs_library_correlation',max(cs)[0] if cs else None,sorted(cs,reverse=True)[:4]); print('period',F.index.min().date(),F.index.max().date())
