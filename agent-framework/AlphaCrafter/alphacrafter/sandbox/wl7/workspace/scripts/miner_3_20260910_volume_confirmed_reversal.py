import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 try:d=get_index_daily_data(s,3000)
 except: d=None
 if d is None:
  try:d=get_stock_daily_data(s,3000)
  except:d=None
 if d is None or len(d)==0:return None
 d=d.copy();d.date=pd.to_datetime(d.date).dt.normalize();return d.drop_duplicates('date').set_index('date').sort_index()
D={s:fetch(s) for s in U};D={s:d for s,d in D.items() if d is not None}
# Volume-confirmed short-horizon reversal: negative 3d return is stronger when
# turnover/volume is unusually high, with robust log-volume z score.
rows=[]
for s,d in D.items():
 x=d[['close','volume']].replace([np.inf,-np.inf],np.nan).dropna(); r=x.close.pct_change(3); lv=np.log1p(x.volume); z=(lv-lv.rolling(20,min_periods=10).median())/(lv.rolling(20,min_periods=10).std()+1e-8); f=-r*(1+z.clip(-2,2).clip(lower=0)); fr=x.close.shift(-1)/x.close-1
 rows.append(pd.DataFrame({'f':f,'fr':fr},index=x.index).dropna().reset_index())
R=pd.concat(rows);obs=[]
for dt,g in R.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:obs.append(g.f.corr(g.fr,method='spearman'))
a=pd.Series(obs).dropna();print('assets',len(D),'dates',len(a),'avg_names',R.groupby('date').size().mean(),'daily IC ICIR hit',a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(252),(a>0).mean())
for h in [5,10,20]:
 q=[]
 for s,d in D.items():
  x=d[['close','volume']].replace([np.inf,-np.inf],np.nan).dropna();r=x.close.pct_change(3);lv=np.log1p(x.volume);z=(lv-lv.rolling(20,min_periods=10).median())/(lv.rolling(20,min_periods=10).std()+1e-8);f=-r*(1+z.clip(-2,2).clip(lower=0));q.append(pd.DataFrame({'f':f,'fr':x.close.shift(-h)/x.close-1}).dropna().reset_index())
 q=pd.concat(q);v=[]
 for _,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:v.append(g.f.corr(g.fr,method='spearman'))
 v=pd.Series(v);print(h,'dates',len(v),'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1)*np.sqrt(252))
for y in range(2020,2027):
 v=[]
 for dt,g in R.groupby('date'):
  if dt.year==y and len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:v.append(g.f.corr(g.fr,method='spearman'))
 v=pd.Series(v);print('regime',y,len(v),v.mean(),v.mean()/v.std(ddof=1)*np.sqrt(252))
print('coverage_dates',len(a)/len(pd.date_range(R.date.min(),R.date.max())))
