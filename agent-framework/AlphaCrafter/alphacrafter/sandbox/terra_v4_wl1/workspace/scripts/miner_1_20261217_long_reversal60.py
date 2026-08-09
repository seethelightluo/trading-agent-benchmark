import pandas as pd, numpy as np, json
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().ffill(); P=P[P.index<=cut]
R=P.pct_change(); vol=R.rolling(60,min_periods=40).std(); F=(-R.rolling(60,min_periods=40).sum()/vol.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan)
F.to_csv('scripts/miner_1_20261217_long_reversal60_signal.csv',index_label='date')
rows=[]
for h in [1,5,10]:
 Y=P.shift(-h)/P-1
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: rows.append((h,dt,spearmanr(z.f,z.y).statistic,len(z)))
for h in [1,5,10]:
 a=pd.DataFrame([x[2:] for x in rows if x[0]==h],index=[x[1] for x in rows if x[0]==h],columns=['ic','n']).ic
 print('H',h,'dates',len(a),'avg_names',round(np.mean([x[3] for x in rows if x[0]==h]),2),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1),8),'hit',round((a>0).mean(),4))
 if h==1: print('regimes',a.groupby(a.index.year).mean().round(6).to_dict())
print('coverage',round(F.notna().sum().sum()/F.size,6),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
corr=[]
for p in Path('scripts').glob('*signal.csv'):
 try:
  x=pd.read_csv(p,index_col=0,parse_dates=True).reindex(F.index).reindex(columns=U); c=F.stack().corr(x.stack())
  if pd.notna(c) and p.name!='miner_1_20261217_long_reversal60_signal.csv': corr.append((abs(c),p.name,float(c)))
 except Exception: pass
print('max_abs_library_correlation',round(max(corr)[0],6) if corr else None, sorted(corr,reverse=True)[:5])
print('period',F.index.min().date(),F.index.max().date(),'assets',len(U))
