import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; B='../persistent/stock_data'
for mode in ['mul','inv','log']:
 xx=[]
 for s in U:
  d=pd.read_csv(f'{B}/{s}.csv',parse_dates=['date']).sort_values('date'); d['r3']=d.close.pct_change(3); d['rd']=d.close.pct_change(); d['v']=d.rd.rolling(30,min_periods=15).std(); q=d.volume.rolling(10,min_periods=5).mean(); d['vs']=(d.volume/q).clip(.5,2); d['fwd']=d.close.shift(-1)/d.close-1;d['s']=s;xx.append(d[['date','r3','v','vs','fwd','s']])
 x=pd.concat(xx);x['m']=x.groupby('date').r3.transform('median'); base=-(x.r3-x.m)/x.v
 if mode=='mul': x['f']=base*x.vs
 elif mode=='inv':x['f']=base/x.vs
 else:x['f']=base*np.log(x.vs)
 a=[]
 for _,g in x.groupby('date'):
  g=g.dropna(subset=['f','fwd'])
  if len(g)>=8:a.append(spearmanr(g.f,g.fwd).statistic)
 a=np.array(a);print(mode,len(a),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0))
