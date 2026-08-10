import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; B='../persistent/stock_data'
for h,vw in [(1,10),(1,30),(2,10),(2,20),(2,30),(3,10),(3,20),(4,10),(4,20),(5,10),(5,20),(5,30)]:
 a=[]; ds=[]; allx=[]
 for s in U:
  d=pd.read_csv(f'{B}/{s}.csv',parse_dates=['date']).sort_values('date'); d['r']=d.close.pct_change(); d['v']=d.r.rolling(vw,min_periods=max(8,vw//2)).std(); d['fwd']=d.close.shift(-h)/d.close-1; d['s']=s; allx.append(d[['date','r','v','fwd','s']])
 x=pd.concat(allx); x['m']=x.groupby('date').r.transform('median'); x['f']=-(x.r-x.m)/x.v
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['f','fwd'])
  if len(g)>=8: a.append(spearmanr(g.f,g.fwd).statistic); ds.append(dt)
 a=np.array(a); print(h,vw,len(a),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6),round(np.mean(a>0),4))
