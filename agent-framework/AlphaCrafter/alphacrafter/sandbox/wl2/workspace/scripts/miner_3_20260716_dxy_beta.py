import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2026-07-15')
def load(s, macro=False):
 p=('../persistent/index_data/' if macro else '../persistent/stock_data/')+s+'.csv'
 d=pd.read_csv(p); d['date']=pd.to_datetime(d['date']); d=d[(d.date<=END)].sort_values('date').drop_duplicates('date',keep='last')
 return d.set_index('date')['close'].astype(float).sort_index()
M=load('DXY',True).pct_change(); R={s:load(s).pct_change() for s in U}
def neg_beta(asset, macro):
 z=pd.concat([asset.rename('a'),macro.rename('m')],axis=1,join='inner').dropna()
 x=z.a.to_numpy(); y=z.m.to_numpy(); out=np.full(len(z),np.nan)
 for i in range(44,len(z)):
  lo=i-59; xx=x[lo:i+1]; yy=y[lo:i+1]; vy=np.var(yy,ddof=1)
  if vy>1e-16: out[i]=-np.cov(xx,yy,ddof=1)[0,1]/vy
 return pd.Series(out,index=z.index)
def observations(h):
 rows=[]
 for s in U:
  z=pd.concat([R[s].rename('a'),M.rename('m')],axis=1,join='inner').dropna(); f=neg_beta(R[s],M).reindex(z.index)
  a=z.a.to_numpy(); dates=z.index
  for i in range(len(z)-h):
   if np.isfinite(f.iloc[i]): rows.append((dates[i],s,f.iloc[i],a[i+1:i+h+1].sum()))
 return pd.DataFrame(rows,columns=['date','s','f','fr'])
base=sum(len(x.dropna()) for x in R.values())
for h in [1,5,10,20]:
 df=observations(h); ic=[]; n=[]
 for dt,g in df.groupby('date'):
  if len(g)>=8:
   v=spearmanr(g.f,g.fr).statistic
   if np.isfinite(v): ic.append(v); n.append(len(g))
 a=np.asarray(ic); print('horizon',h,'dates',len(a),'avg_names',round(np.mean(n),2),'coverage',round(len(df)/base,4),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1),6),'hit',round(np.mean(a>0),4))
x=observations(1); vals=[]
for dt,g in x.groupby('date'):
 if len(g)>=8: vals.append((dt,spearmanr(g.f,g.fr).statistic))
v=pd.DataFrame(vals,columns=['date','ic']); v['regime']=pd.cut(v.date.dt.year,[2019,2021,2023,2024,2026],labels=['2020-21','2022-23','2024','2025-26'])
print('regimes',v.groupby('regime',observed=True).ic.agg(['mean','count']).round(6).to_dict('index'))
print('factor negative_rolling_dxy_beta_60d')
