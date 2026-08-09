import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-12-17')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().loc[:END]
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').close.reindex(P.index).ffill()
# lagged DXY 20d trend, standardized only with prior observations
tr=dxy.shift(1).pct_change(20); z=(tr-tr.rolling(252,min_periods=60).mean())/tr.rolling(252,min_periods=60).std(); gate=(z>0).astype(float)
r=P.pct_change(7).shift(1); vol=P.pct_change().rolling(20,min_periods=10).std().shift(1)
med=r.median(axis=1); base=-(r.sub(med,axis=0)).div(vol)
f=base.where(gate>0) # conditioned signal; dates with gate only
f.to_csv('scripts/miner_3_20261217_dxy_conditioned_reversal_signal.csv',index_label='date')
for h in [1,5,10]:
 y=P.shift(-h).div(P)-1; vals=[]; ns=[]
 for d in P.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1: vals.append(spearmanr(q.f,q.y).statistic);ns.append(len(q))
 a=np.array(vals);print('H',h,'dates',len(a),'avg_n',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
print('coverage',round(f.notna().sum().sum()/f.size,5),'active_dates',f.notna().any(axis=1).sum(),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
print('period',P.index.min().date(),P.index.max().date())
