import numpy as np, pandas as pd, glob
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
series={}
for s in U:
 fs=glob.glob('../persistent/stock_data/'+s+'.csv')
 if fs:
  d=pd.read_csv(fs[0]); d['date']=pd.to_datetime(d['date']); d=d.set_index('date').sort_index()
  if len(d)>300: series[s]=d
rows=[]
for s,d in series.items():
 c=d.close.astype(float); h=d.high.astype(float); l=d.low.astype(float)
 loc=((c-l)/(h-l).replace(0,np.nan)-.5).rolling(5,min_periods=3).mean()
 raw=-c.pct_change(10)*(1+1.5*loc.abs())/(c.pct_change().rolling(30,min_periods=15).std()*np.sqrt(10)+1e-8)
 rows.append(pd.DataFrame({'date':c.index,'symbol':s,'factor':raw.shift(1),'close':c}))
x=pd.concat(rows,ignore_index=True)
for H in [5,10,20,30]:
 z=x.sort_values(['symbol','date']).copy(); z['fwd']=z.groupby('symbol').close.shift(-H)/z.close-1; v=[]; ns=[]
 for _,g in z.groupby('date'):
  g=g.dropna(subset=['factor','fwd'])
  if len(g)>=8: v.append(g.factor.corr(g.fwd,method='spearman')); ns.append(len(g))
 q=pd.Series(v).dropna(); print('H',H,'dates',len(q),'N',round(np.mean(ns),2),'IC',round(q.mean(),8),'ICIR',round(q.mean()/(q.std(ddof=1)+1e-12),6),'hit',round((q>0).mean(),4))
z['rank']=z.groupby('date').factor.rank(pct=True); z['rl']=z.groupby('symbol').rank.shift(1) if False else z.groupby('symbol')['rank'].shift(1)
print('symbols',len(series),'dates',x.date.nunique(),'coverage',x.factor.notna().mean(),'turnover',((z['rank']-z.rl).abs().dropna()).mean())
