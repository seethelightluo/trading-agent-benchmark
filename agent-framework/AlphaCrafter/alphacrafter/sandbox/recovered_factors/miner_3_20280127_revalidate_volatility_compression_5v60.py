"""miner_3 scheduled revalidation: short-to-long volatility compression, as of 2028-01-26."""
import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2028-01-26')
def read(a,field='close',base='../persistent/stock_data/'):
 d=pd.read_csv(base+a+'.csv',parse_dates=['date']).query('date<=@E').drop_duplicates('date').set_index('date').sort_index(); return pd.to_numeric(d[field],errors='coerce')
P=pd.DataFrame({a:read(a) for a in A}); R=P.pct_change(fill_method=None); v5=R.rolling(5,min_periods=4).std(); v20=R.rolling(20,min_periods=15).std(); v60=R.rolling(60,min_periods=45).std(); F=-v5/(v60+1e-12)
def stats(x):
 sd=x.std(ddof=1); return (x.mean(),x.mean()/sd if sd else np.nan,(x>0).mean(),len(x),sd/np.sqrt(len(x)) if len(x) else np.nan)
print('REVALIDATION volatility_compression_5v60; visible through',E.date(),'assets',len(A))
ics={}
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; z=[]; ns=[]
 for d in P.index:
  q=pd.concat([F.loc[d].rename('f'),fw.loc[d].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:z.append((d,q.f.corr(q.r,method='spearman')));ns.append(len(q))
 x=pd.Series(dict(z));ics[h]=x; a=stats(x); print(f'h={h} dates={a[3]} IC={a[0]:.6f} ICIR={a[1]:.6f} hit={a[2]:.6f} instruments={np.mean(ns):.2f} se={a[4]:.6f}')
for label,mask in [('2020_21',ics[1].index<'2022-01-01'),('2022_23',(ics[1].index>='2022-01-01')&(ics[1].index<'2024-01-01')),('2024_25',(ics[1].index>='2024-01-01')&(ics[1].index<'2026-01-01')),('2026_27',ics[1].index>='2026-01-01')]:
 x=ics[1][mask];a=stats(x);print('REGIME',label,'dates',a[3],'IC',f'{a[0]:.6f}' if a[3] else None,'ICIR',f'{a[1]:.6f}' if a[3]>1 else None,'hit',f'{a[2]:.6f}' if a[3] else None)
r=F.rank(axis=1,pct=True);turn=[]
for i in range(1,len(r)):
 q=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(q)>=8:turn.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('coverage',f'{F.notna().mean().mean():.6f}','valid_cells',int(F.notna().sum().sum()),'rank_turnover',f'{np.mean(turn):.6f}')
print('DECISION selected_same_horizon=1d passes=',abs(stats(ics[1])[0])>=.007 and abs(stats(ics[1])[1])>=.084)
