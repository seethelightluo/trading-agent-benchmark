import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-11-03');P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv');x.date=pd.to_datetime(x.date);P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); z=pd.read_csv('../persistent/index_data/DXY.csv');z.date=pd.to_datetime(z.date);z=z[z.date<=END].set_index('date').close.sort_index().reindex(px.index).ffill(); dr=z.pct_change().shift(1); base=-px.pct_change().shift(1); th=dr.abs().rolling(60,min_periods=30).quantile(.75).shift(1); shock=(dr.abs()>th); fwd=px.shift(-1)/px-1
# On lagged dollar shocks, test prior return reversal; non-shock dates excluded, not multiplied uniformly.
def calc(mask):
 a=[];ns=[]
 for i,d in enumerate(px.index):
  if not bool(mask.iloc[i]):continue
  g=pd.DataFrame({'s':base.loc[d],'f':fwd.loc[d]}).dropna()
  if len(g)>=8 and g.s.nunique()>1:
   q=spearmanr(g.s,g.f).statistic
   if np.isfinite(q):a.append(q);ns.append(len(g))
 a=np.array(a);return len(a),round(np.mean(ns),2),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6),round((a>0).mean(),4)
print('end',px.index.max().date(),'shock',calc(shock),'nonshock',calc(~shock)); y=px.index.year
for q,m in [('2020-22',(y>=2020)&(y<=2022)),('2023-25',(y>=2023)&(y<=2025)),('2026',y==2026),('2027',y==2027),('last180',px.index>=END-pd.Timedelta(days=180))]:print(q,'shock',calc(shock&m))
out=base.where(shock).stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20271104_dxy_shock_reversal_signal.csv',index=False)
print('coverage',int(base.where(shock).notna().sum().sum()),'of',base.size)
