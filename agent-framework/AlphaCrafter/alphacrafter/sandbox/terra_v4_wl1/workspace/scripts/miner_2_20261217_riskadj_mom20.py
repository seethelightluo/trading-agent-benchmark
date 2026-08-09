import numpy as np,pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().ffill(); P=P[P.index<=cut]; R=P.pct_change()
F=R.rolling(20,min_periods=15).sum()/R.rolling(20,min_periods=15).std().replace(0,np.nan); F=F.clip(-5,5); F.to_csv('scripts/miner_2_20261217_riskadj_mom20_signal.csv',index_label='date')
for h in [1,5,10]:
 Y=P.pct_change(h).shift(-h); vals=[];ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: vals.append((dt,spearmanr(z.f,z.y).statistic));ns.append(len(z))
 a=pd.DataFrame(vals,columns=['date','ic']).set_index('date').ic
 print('H',h,'dates',len(a),'avg_names',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
 if h==1: print('regimes',a.groupby(a.index.year).mean().to_dict())
print('coverage',F.notna().sum().sum()/F.size,'turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
cs=[]
for p in Path('scripts').glob('*signal.csv'):
 try:
  x=pd.read_csv(p,index_col=0,parse_dates=True).reindex(F.index).reindex(columns=U); c=F.stack().corr(x.stack())
  if pd.notna(c):cs.append((abs(c),p.name,c))
 except:pass
print('max_abs_library_correlation',max(cs)[0] if cs else None)
