import pandas as pd,numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17')
P=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill(); P=P[P.index<=cut]; R=P.pct_change()
# Trend-reversal blend: short-term mean reversion tempered by medium-term trend.
short=-R.rolling(3,min_periods=3).sum(); med=R.rolling(20,min_periods=15).sum();
def cs(x):
 m=x.mean(axis=1); sd=x.std(axis=1).replace(0,np.nan); return x.sub(m,axis=0).div(sd,axis=0)
F=cs(short)+0.35*cs(med); F=F.replace([np.inf,-np.inf],np.nan); F.to_csv('scripts/miner_3_20261217_trendreversal_blend_signal.csv',index_label='date')
for h in [1,5,10]:
 Y=P.pct_change(h).shift(-h); vals=[]; ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.f,z.y).statistic); ns.append(len(z))
 a=pd.Series(vals); print('H',h,'dates',len(a),'avg_names',round(np.mean(ns),2),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1),8),'hit',round((a>0).mean(),4))
print('coverage',round(F.notna().sum().sum()/F.size,6),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),6)); print('period',F.index.min().date(),F.index.max().date(),'assets',len(U))
# correlations against prior signal artifacts, excluding self
corr=[]
for p in Path('scripts').glob('*signal.csv'):
 if p.name.endswith('trendreversal_blend_signal.csv'): continue
 try:
  x=pd.read_csv(p,index_col=0,parse_dates=True).reindex(F.index).reindex(columns=U); c=F.stack().corr(x.stack())
  if pd.notna(c): corr.append((abs(c),p.name,c))
 except Exception: pass
print('max_abs_library_correlation',max(corr)[0] if corr else None)
