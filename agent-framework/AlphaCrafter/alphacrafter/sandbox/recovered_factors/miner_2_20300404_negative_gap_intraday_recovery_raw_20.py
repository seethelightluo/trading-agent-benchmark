"""Single idea: negative-gap intraday recovery, raw volatility-normalized 20-day severity-weighted score (residualization omitted because calendars make cross-sectional controls unavailable)."""
import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];E=pd.Timestamp('2030-04-03')
def load(a,c='close'):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index();return pd.to_numeric(d[c],errors='coerce')
P=pd.DataFrame({a:load(a) for a in A});O=pd.DataFrame({a:load(a,'open') for a in A});R=P.pct_change(fill_method=None);V=R.rolling(20,min_periods=15).std()
gap=O/P.shift(1)-1;sev=(-gap/(V.shift(1)+1e-12)).clip(0,4);raw= (P/O-1).mul(sev).rolling(20,min_periods=10).sum().div(sev.rolling(20,min_periods=10).sum().replace(0,np.nan),axis=0).div(V+1e-12)
# Cross-calendar alignment requires at least eight contemporaneously quoted assets.
F=raw
print('FACTOR negative_gap_intraday_recovery_volnorm_20 visible_through',E.date(),'assets',len(A),'factor_cells',int(F.notna().sum().sum()))
ics={}
for h in [1,5,10,20]:
 out=[];ns=[];fw=P.shift(-h)/P-1
 for t in P.index[P.index<=E]:
  q=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:out.append((t,q.f.corr(q.r,method='spearman')));ns.append(len(q))
 x=pd.Series([v for t,v in out],index=pd.DatetimeIndex([t for t,v in out]),dtype=float);ics[h]=x;sd=x.std(ddof=1)
 print(f'h={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} instruments={np.mean(ns):.2f} se={sd/np.sqrt(len(x)):.6f}')
for n,lo,hi in [('2020_21','2020-01-01','2022-01-01'),('2022_23','2022-01-01','2024-01-01'),('2024_25','2024-01-01','2026-01-01'),('2026_27','2026-01-01','2028-01-01'),('2028_30','2028-01-01','2031-01-01')]:
 x=ics[10][(ics[10].index>=lo)&(ics[10].index<hi)];print('regime',n,'dates',len(x),'IC',f'{x.mean():.6f}','ICIR',f'{x.mean()/x.std(ddof=1):.6f}' if len(x)>1 else 'nan','hit',f'{(x>0).mean():.4f}')
r=F.rank(axis=1,pct=True);to=[]
for j in range(1,len(r)):
 q=pd.concat([r.iloc[j-1],r.iloc[j]],axis=1).dropna()
 if len(q)>=8:to.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('coverage',f'{F.notna().mean().mean():.6f}','turnover',f'{np.mean(to):.6f}')
print('NOVELTY: full historical panels for every admitted library signal were not reconstructible in this cycle; max_abs_library_correlation missing => binding admission fails.')
