import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-30')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut] for s in U}; c=pd.DataFrame({s:x.close for s,x in P.items()}).ffill(); v=pd.DataFrame({s:x.volume for s,x in P.items()}).ffill(); r=c.pct_change(); vr=np.log((v.rolling(5,min_periods=5).mean()+1e-9)/(v.rolling(60,min_periods=30).mean()+1e-9)); vr=vr.sub(vr.mean(axis=1),axis=0); f=(r.rolling(5).sum()*vr).shift(1)
for h in [5,10,20]:
 a=[];n=[]
 for i in range(len(c)-h):
  q=pd.concat([f.iloc[i].rename('f'),(c.iloc[i+h]/c.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:a.append(spearmanr(q.f,q.y).statistic);n.append(len(q))
 a=np.array(a);print(h,len(a),np.mean(n),np.mean(a),np.mean(a)/np.std(a,ddof=1),np.mean(a>0))
print('turnover',((f.rank(pct=True)-f.rank(pct=True).shift()).abs().stack().groupby(level=0).mean().dropna().mean()))
