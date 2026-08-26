import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; frames={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d['date']=pd.to_datetime(d['date']); d=d.sort_values('date').set_index('date'); frames[s]=d
rows=[]
for s,d in frames.items():
 r=d.close.pct_change(); z=(d.high-d.low)/d.close.shift(1); shock=z/(z.rolling(30,min_periods=20).median()+1e-12); v=r.rolling(30,min_periods=20).std(); f=-(r.shift(1))*shock.shift(1)/(v.shift(1)+1e-12); y=d.close.shift(-10)/d.close-1
 rows.append(pd.DataFrame({'date':d.index,'symbol':s,'factor':f.values,'fwd':y.values}).dropna())
panel=pd.concat(rows,ignore_index=True); a=[]
for dt,g in panel.groupby('date'):
 if len(g)>=8 and g.factor.nunique()>1 and g.fwd.nunique()>1:a.append(spearmanr(g.factor,g.fwd).statistic)
ic=np.array(a); print({'dates':len(ic),'avg_instruments':panel.groupby('date').size().mean(),'instruments':len(frames),'mean_daily_paper_ic':ic.mean(),'daily_paper_icir':ic.mean()/ic.std(ddof=1)*np.sqrt(252),'hit_ratio':(ic>0).mean()})
for h in [1,5,10,20]:
 qs=[]
 for s,d in frames.items():
  r=d.close.pct_change(); z=(d.high-d.low)/d.close.shift(1); sh=z/(z.rolling(30,min_periods=20).median()+1e-12); v=r.rolling(30,min_periods=20).std(); f=-(r.shift(1))*sh.shift(1)/(v.shift(1)+1e-12); y=d.close.shift(-h)/d.close-1; qs.append(pd.DataFrame({'date':d.index,'f':f.values,'y':y.values}).dropna())
 q=pd.concat(qs,ignore_index=True); aa=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:aa.append(spearmanr(g.f,g.y).statistic)
 print('horizon',h,'ic',np.mean(aa),'n',len(aa))
