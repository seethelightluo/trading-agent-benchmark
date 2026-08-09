import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17')
base='../persistent/stock_data'; macro='../persistent/index_data'
P=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index().close for s in U}).sort_index(); P=P[P.index<=cut]
v=pd.read_csv(f'{macro}/VIX.csv',parse_dates=['date']).set_index('date').sort_index().close.astype(float); v=v[v.index<=cut]
v5=v.pct_change(5).reindex(P.index).ffill().clip(-.5,.5)
f=(-P.pct_change(5).mul(1+v5,axis=0)).where(v5>0,-P.pct_change(5)*.5)
print('instruments',len(P.columns),'period',P.index.min(),P.index.max())
for h in [1,5,10]:
 y=P.shift(-h)/P-1; vals=[]; ns=[]
 for d in P.index:
  z=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8: vals.append(z.f.corr(z.y,method='spearman'));ns.append(len(z))
 a=pd.Series(vals).dropna();print('H',h,'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
for label,mask in [('pre2024',P.index<'2024-01-01'),('2024+',P.index>='2024-01-01'),('2026+',P.index>='2026-01-01')]:
 y=P.shift(-1)/P-1; a=[]
 for d in P.index[mask]:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=pd.Series(a).dropna();print(label,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1))
print('coverage',f.notna().sum().sum()/(f.shape[0]*f.shape[1]),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
print('signal_artifact','scripts/miner_1_20261217_macro_conditioned_reversal_signal.csv')
