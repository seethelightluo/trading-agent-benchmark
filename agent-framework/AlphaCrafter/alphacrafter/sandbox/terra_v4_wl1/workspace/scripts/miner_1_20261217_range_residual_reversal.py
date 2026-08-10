import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; B='../persistent/stock_data'
for w in [10,20,30,60]:
 xx=[]
 for s in U:
  d=pd.read_csv(f'{B}/{s}.csv',parse_dates=['date']).sort_values('date'); d['r']=d.close.pct_change(); d['atr']=(d.high-d.low)/d.close; d['v']=d.atr.rolling(w,min_periods=max(8,w//2)).mean(); d['fwd']=d.close.shift(-1)/d.close-1; d['s']=s; xx.append(d[['date','r','v','fwd','s']])
 x=pd.concat(xx); x['m']=x.groupby('date').r.transform('median'); x['f']=-(x.r-x.m)/x.v
 a=[]
 for _,g in x.groupby('date'):
  g=g.dropna(subset=['f','fwd'])
  if len(g)>=8:a.append(spearmanr(g.f,g.fwd).statistic)
 a=np.array(a); print(w,len(a),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0))
