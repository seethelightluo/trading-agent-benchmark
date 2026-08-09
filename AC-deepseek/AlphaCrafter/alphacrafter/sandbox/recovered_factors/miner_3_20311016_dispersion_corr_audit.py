import pandas as pd,numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in assets}; px=pd.concat(D,axis=1).sort_index().ffill(); r=px.pct_change(); vol=r.rolling(20,min_periods=15).std(); disp=r.sub(r.median(axis=1),axis=0).abs().mean(axis=1); hi=disp.rolling(120,min_periods=60).rank(pct=True)>=.7
cand=-(px.pct_change(3)/vol).where(hi.values[:,None])
# signal panel: admitted-factor-style price/macro proxies, evaluated only where candidate exists
p={'risk_adj_trend20':px.pct_change(20)/r.rolling(20,min_periods=15).std(),'volnorm_rev5':-px.pct_change(5)/r.rolling(5,min_periods=4).std(),'inv_vol':-vol,'trend60':px.pct_change(60),'skew40':-r.rolling(40,min_periods=30).skew(),'kurt40':-r.rolling(40,min_periods=30).kurt(),'peer_corr':-r.rolling(40,min_periods=30).corr(r.mean(axis=1)).groupby(level=0).mean() if False else -r.rolling(40,min_periods=30).corr()}
# only matrix-compatible signals
mx=[]
for k,v in list(p.items())[:6]:
 if v.ndim==2:
  a=cand.stack().rename('c').to_frame().join(v.stack().rename(k)).dropna();
  if len(a)>100: mx.append((k,abs(spearmanr(a.c,a[k]).statistic),len(a)))
print('proxy correlations',sorted(mx,key=lambda x:-x[1]))
