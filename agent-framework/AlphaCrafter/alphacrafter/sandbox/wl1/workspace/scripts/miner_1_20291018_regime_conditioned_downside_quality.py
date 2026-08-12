import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2029-10-17')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut] for s in U}
idx=sorted(set().union(*[set(x.index) for x in P.values()])); px=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill(); r=px.pct_change()
# Regime-conditioned downside-quality momentum: lagged 20d return/downside vol,
# reverse the signal in broad weak tape where short-term continuation is unreliable.
ret=r.rolling(20,min_periods=15).sum(); down=r.clip(upper=0).rolling(40,min_periods=30).std(); breadth=(ret>0).mean(axis=1)
reg=np.where(breadth>=0.50,1.0,-1.0)
f=(ret/(down*np.sqrt(20)+1e-8)).mul(reg,axis=0).shift(1)
print('factor=lagged downside-quality momentum, sign-reversed when 20d cross-sectional breadth < 50%')
for h in [5,10,20]:
 I=[];ds=[];ns=[]
 for i in range(len(px)-h):
  q=pd.concat([f.iloc[i].rename('f'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   I.append(spearmanr(q.f,q.y).statistic);ds.append(px.index[i]);ns.append(len(q))
 a=np.array(I); d=pd.DatetimeIndex(ds)
 print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lab,m in [('2020-25',(d<pd.Timestamp('2026-01-01'))),('2026+',d>=pd.Timestamp('2026-01-01')),('2028+',d>=pd.Timestamp('2028-01-01')),('2029YTD',d>=pd.Timestamp('2029-01-01'))]:
  z=a[m]; print(' ',lab,'n',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else np.nan)
 if h==10:
  f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20291018_regime_conditioned_downside_quality_signal.csv',index=False)
print('coverage',round(f.notna().sum().sum()/f.size,6),'rank_turnover',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),6),'assets',len(U),'dates',len(px))
