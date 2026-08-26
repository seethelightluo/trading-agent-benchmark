import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def calc(d,h):
 intr=d.close/d.open-1; vol=d.close.pct_change().rolling(30,min_periods=20).std(); f=-(intr.rolling(5,min_periods=5).sum()).shift(1)/(vol.shift(1)+1e-12); y=d.close.shift(-h)/d.close-1
 return pd.DataFrame({'date':d.index,'f':f.to_numpy(),'y':y.to_numpy()}).dropna()
frames=[]; total=0
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); total+=len(d); d['date']=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date'); frames.append(calc(d,10))
q=pd.concat(frames,ignore_index=True); z=[]
for dt,g in q.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:z.append(spearmanr(g.f,g.y).statistic)
ic=np.asarray(z)
print({'dates':len(ic),'avg_instruments':q.groupby('date').size().mean(),'instruments':len(frames),'coverage':len(q)/total,'mean_daily_paper_ic':float(ic.mean()),'daily_paper_icir':float(ic.mean()/ic.std(ddof=1)*np.sqrt(252)),'hit_ratio':float((ic>0).mean())})
for h in [1,5,10,20]:
 q=pd.concat([calc(pd.read_csv('../persistent/stock_data/'+s+'.csv').assign(date=lambda x:pd.to_datetime(x.date)).sort_values('date').set_index('date'),h) for s in U if os.path.exists('../persistent/stock_data/'+s+'.csv')],ignore_index=True); z=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:z.append(spearmanr(g.f,g.y).statistic)
 print('horizon',h,'ic',float(np.mean(z)),'n',len(z))
