import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300:
  try:d=get_index_daily_data(s,4000)
  except Exception:d=None
 if d is not None:P[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(P).sort_index().ffill();r=np.log(p).diff()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.astype(float).reindex(r.index).ffill()
vp=v.rolling(120,min_periods=40).rank(pct=True)
# 5-day contrarian signal, selectively intensified during elevated VIX regimes.
F=-r.rolling(5,min_periods=5).sum().mul((0.55+1.10*vp),axis=0).replace([np.inf,-np.inf],np.nan)
rows=[]
for t in F.index:
 j=r.index.searchsorted(t,side='right'); k=j+0
 if j>=len(r):continue
 z=pd.concat([F.loc[t],r.iloc[j]],axis=1).dropna()
 if len(z)>=8:rows.append((t,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
x=pd.DataFrame(rows,columns=['date','ic','n']);a=x.ic.dropna();print('dates',x.date.nunique(),'instruments',len(U),'obs',len(x),'avg_n',round(x.n.mean(),3),'coverage',round(F.notna().stack().mean(),5),'rank_turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),5));print('1d IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for h in [3,5,10]:
 q=[]
 for t in F.index:
  j=r.index.searchsorted(t,side='right');k=j+h-1
  if j>=len(r) or k>=len(r):continue
  z=pd.concat([F.loc[t],r.iloc[j:k+1].sum()],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print(str(h)+'d IC',round(np.nanmean(q),6),'obs',len(q))
for lo,hi in [(2020,2022),(2023,2025),(2026,2027),(2028,2030),(2031,2031)]:
 q=x[x.date.dt.year.between(lo,hi)].ic;print('REG',lo,hi,round(q.mean(),6) if len(q) else None,round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None,len(q))
F.to_csv('scripts/miner_1_20310320_vix_conditioned_reversal_signal.csv')
