import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 try: D[s]=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date')
 except: pass

def run(name,fun):
 rec=[]
 for s,x in D.items():
  f=fun(x)
  for i,dt in enumerate(x.index[:-1]):
   if pd.notna(f.iloc[i]): rec.append((dt,s,float(f.iloc[i]),float(x.close.iloc[i+1]/x.close.iloc[i]-1)))
 a=pd.DataFrame(rec,columns=['d','s','f','y']); a=a[a.d>='2026-07-16']; z=[]
 for d,g in a.groupby('d'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:z.append(spearmanr(g.f,g.y).statistic)
 z=np.array(z); print(name,'dates',len(z),'avgN',a.groupby('d').size().mean(),'coverage',a.s.nunique()/15,'IC',np.nanmean(z),'ICIR',np.nanmean(z)/np.nanstd(z,ddof=1),'hit',np.mean(z>0),'turnover','n/a')
run('clv_1d',lambda x: -(2*(x.close-x.low)/(x.high-x.low).replace(0,np.nan)-1))
run('reversal_5d',lambda x: -x.close.pct_change(5))
run('reversal_3d',lambda x: -x.close.pct_change(3))
run('risk_adj_mom20',lambda x: x.close.pct_change(20)/(x.close.pct_change().rolling(20).std()+1e-12))
run('range_eff_20',lambda x: (x.high-x.low).rolling(20).mean()/(x.close.pct_change().abs().rolling(20).sum()+1e-12))
