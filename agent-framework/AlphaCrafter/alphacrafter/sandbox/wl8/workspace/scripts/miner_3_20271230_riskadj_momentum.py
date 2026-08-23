import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-12-29')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); v=r.rolling(20,min_periods=15).std().shift(1)
sig=(r.rolling(10).sum().shift(1)/v).clip(-10,10); fwd=px.shift(-1)/px-1
def calc(mask):
 vals=[]; ns=[]
 for d in px.index[pd.Series(mask,index=px.index).fillna(False)]:
  g=pd.DataFrame({'s':sig.loc[d],'f':fwd.loc[d]}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(g)>=8 and g.s.nunique()>1:
   z=spearmanr(g.s,g.f).statistic
   if np.isfinite(z): vals.append(z); ns.append(len(g))
 a=np.array(vals); return len(a),round(np.mean(ns),2),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6),round((a>0).mean(),4)
print('end',px.index.max().date(),'all',calc(np.ones(len(px),bool))); y=px.index.year
for q,m in [('2020-22',y<=2022),('2023-25',(y>=2023)&(y<=2025)),('2026',y==2026),('2027',y==2027),('last180',px.index>=END-pd.Timedelta(days=180))]: print(q,calc(m))
print('coverage',round(sig.notna().sum().sum()/sig.size,4)); sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20271230_riskadj_momentum_signal.csv',index=False)
