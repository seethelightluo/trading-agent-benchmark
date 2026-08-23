import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-12-29')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); d=pd.read_csv('../persistent/index_data/DXY.csv'); d.date=pd.to_datetime(d.date); d=d[d.date<=END].set_index('date').close.sort_index().reindex(px.index).ffill()
dr=d.pct_change(); z=(dr.rolling(5).sum()/dr.rolling(5).std().rolling(60,min_periods=30).std()).shift(1).clip(-2,2)
# match stored conceptual factor, use amplification
sig=-px.pct_change(3).shift(1).mul(1+0.5*z.abs(),axis=0); fwd=px.shift(-1)/px-1
def calc(mask):
 vals=[]; ns=[]
 for dt in px.index[pd.Series(mask,index=px.index).fillna(False)]:
  g=pd.DataFrame({'s':sig.loc[dt],'f':fwd.loc[dt]}).dropna()
  if len(g)>=8 and g.s.nunique()>1:
   q=spearmanr(g.s,g.f).statistic
   if np.isfinite(q): vals.append(q);ns.append(len(g))
 a=np.array(vals); return len(a),round(np.mean(ns),2),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6),round((a>0).mean(),4)
y=px.index.year; print('end',px.index.max().date(),'all',calc(np.ones(len(px),bool)))
for q,m in [('2020-22',y<=2022),('2023-25',(y>=2023)&(y<=2025)),('2026',y==2026),('2027',y==2027),('last180',px.index>=END-pd.Timedelta(days=180))]:print(q,calc(m))
print('coverage',round(sig.notna().sum().sum()/sig.size,4))
