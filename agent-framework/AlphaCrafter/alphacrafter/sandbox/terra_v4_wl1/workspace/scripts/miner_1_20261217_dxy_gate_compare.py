import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
xs=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END][['date','close']]; d['r5']=d.close.pct_change(5); d['y1']=d.close.shift(-1)/d.close-1; xs.append(d.assign(symbol=s))
x=pd.concat(xs); med=x.groupby('date').r5.transform('median'); x['resid']=-(x.r5-med).shift(1)
m=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).sort_values('date'); m['r20']=m.close.pct_change(20); m['trend']=m.r20.shift(1); x=x.merge(m[['date','trend']],on='date',how='left')
for label,gate in [('up',x.trend>0),('down',x.trend<0),('extreme',x.trend.abs()>x.trend.rolling(60,min_periods=30).median())]:
 x['f']=x.resid*gate.astype(float); a=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['f','y1']);
  if len(g)>=8 and g.f.nunique()>1:a.append(spearmanr(g.f,g.y1).statistic)
 a=np.array(a);print(label,len(a),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6),round((a>0).mean(),4))
