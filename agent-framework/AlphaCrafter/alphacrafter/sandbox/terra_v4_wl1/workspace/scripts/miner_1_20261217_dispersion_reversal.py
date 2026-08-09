import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().ffill(); P=P[P.index<=cut]
R=P.pct_change(); r5=R.rolling(5,min_periods=5).sum()
# Dispersion-conditioned reversal. All inputs end at signal date; forward return starts next day.
disp=R.rolling(20,min_periods=15).std().mean(axis=1)
scale=(disp/disp.rolling(120,min_periods=60).median()).clip(0.5,2.0)
F=-r5.mul(scale,axis=0); F=F.sub(F.median(axis=1),axis=0)
F.to_csv('scripts/miner_1_20261217_dispersion_reversal_signal.csv',index_label='date')
for h in [1,5,10]:
 Y=P.pct_change(h).shift(-h); vals=[]; ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),Y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: vals.append(spearmanr(z.f,z.y).statistic); ns.append(len(z))
 a=pd.Series(vals); print('H',h,'dates',len(a),'avg_names',round(np.mean(ns),2),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1),8),'hit',round((a>0).mean(),4))
print('coverage',round(F.notna().sum().sum()/F.size,6),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),6),'period',F.index.min().date(),F.index.max().date())
# regime splits
for label,ix in [('pre2024',F.index<'2024-01-01'),('2024_25',(F.index>='2024-01-01')&(F.index<'2026-01-01')),('online',F.index>='2026-07-16')]:
 Y=P.pct_change().shift(-1); a=[]
 for d in F.index[ix]:
  z=pd.concat([F.loc[d].rename('f'),Y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.f,z.y).statistic)
 a=pd.Series(a); print(label,len(a),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6) if len(a)>1 else None)
cs=[]
for p in Path('scripts').glob('*signal.csv'):
 try:
  x=pd.read_csv(p,index_col=0,parse_dates=True).reindex(F.index).reindex(columns=U); c=F.stack().corr(x.stack())
  if pd.notna(c):cs.append((abs(c),p.name,c))
 except Exception: pass
print('max_abs_library_correlation',round(max(cs)[0],6) if cs else None)
