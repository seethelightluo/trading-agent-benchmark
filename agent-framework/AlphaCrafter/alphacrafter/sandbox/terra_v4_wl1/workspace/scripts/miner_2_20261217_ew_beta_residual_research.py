import pandas as pd,numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().ffill(); P=P[P.index<=cut]
R=P.pct_change(); bench=R.mean(axis=1)
beta=R.ewm(span=40,min_periods=25,adjust=False).cov(bench).div(bench.ewm(span=40,min_periods=25,adjust=False).var(),axis=0)
res=R-beta.mul(bench,axis=0); vol=R.rolling(20,min_periods=15).std()
# Smooth the lagged residual-reversal signal over three completed sessions.
F=(-(res/vol)).replace([np.inf,-np.inf],np.nan).rolling(3,min_periods=3).mean()
F.to_csv('scripts/miner_2_20261217_ew_beta_residual_signal.csv',index_label='date')
for h in [1,5,10]:
 Y=P.pct_change(h).shift(-h); vals=[]; ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   c=spearmanr(z.f,z.y).statistic
   if np.isfinite(c): vals.append((dt,c)); ns.append(len(z))
 a=pd.Series(dict(vals)).sort_index(); print('H',h,'dates',len(a),'avg_names',round(np.mean(ns),2),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1),8),'hit',round((a>0).mean(),4))
 if h==1: print('regimes',a.groupby(a.index.year).mean().round(6).to_dict())
print('coverage',round(F.notna().sum().sum()/F.size,6),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
corr=[]
for p in Path('scripts').glob('*signal.csv'):
 try:
  x=pd.read_csv(p,index_col=0,parse_dates=True).reindex(F.index).reindex(columns=U); c=F.stack().corr(x.stack())
  if pd.notna(c): corr.append((abs(c),p.name,c))
 except: pass
print('max_abs_library_correlation',max(corr)[0] if corr else None, sorted(corr,reverse=True)[:5])
print('period',F.index.min().date(),F.index.max().date(),'assets',len(U))
