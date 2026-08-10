import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; B='../persistent/stock_data'
for lag in [5,10,20]:
 xx=[]
 for s in U:
  d=pd.read_csv(f'{B}/{s}.csv',parse_dates=['date']).sort_values('date'); d['r3']=d.close.pct_change(3); d['rd']=d.close.pct_change(); d['v']=d.rd.rolling(30,min_periods=15).std(); d['volshock']=d.volume/(d.volume.rolling(lag,min_periods=max(3,lag//2)).mean()).replace(0,np.nan); d['fwd']=d.close.shift(-1)/d.close-1; d['s']=s; xx.append(d[['date','r3','v','volshock','fwd','s']])
 x=pd.concat(xx); x['m']=x.groupby('date').r3.transform('median'); x['f']=-(x.r3-x.m)/x.v * (x.volshock.clip(0.5,2)-1).abs().add(1)
 a=[]; ds=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['f','fwd'])
  if len(g)>=8:a.append(spearmanr(g.f,g.fwd).statistic);ds.append(dt)
 a=np.array(a);print(lag,len(a),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0))
