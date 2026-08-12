import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<150: d=get_index_daily_data(s,days=3000)
 if d is not None and len(d)>150:
  z=d.copy(); z['date']=pd.to_datetime(z['date']); P[s]=z.set_index('date')['close'].sort_index()
px=pd.concat(P,axis=1).sort_index().ffill(); r=px.pct_change(); m=r['SPX']
beta=r.rolling(60,min_periods=40).cov(m).div(m.rolling(60,min_periods=40).var(),axis=0)
res=r.sub(beta.mul(m,axis=0),axis=0)
# Longer horizon, gated by cross-sectional positive residual breadth; neutral otherwise.
trend=np.log1p(res.clip(lower=-.99)).rolling(40,min_periods=30).sum().pipe(np.expm1)
vol=res.rolling(40,min_periods=30).std()*np.sqrt(252)
raw=(trend/vol).replace([np.inf,-np.inf],np.nan)
breadth=(res.rolling(20,min_periods=15).mean()>0).mean(axis=1)
gate=(breadth>=0.50).astype(float)*1.0 + (breadth<0.50).astype(float)*0.35
f=raw.mul(gate,axis=0).shift(1)
print('universe',len(U),'loaded',len(P),'dates',len(px))
for h in [1,3,5,10]:
 fr=px.pct_change(h).shift(-h); vals=[]; ns=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   q=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if pd.notna(q): vals.append(q); ns.append(len(a))
 x=pd.Series(vals); print('H',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
# regime breakdown at admission horizon
fr=px.pct_change(10).shift(-10); vals=[]
for dt in f.index:
 a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(a)>=8:
  q=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
  if pd.notna(q): vals.append((dt,q))
x=pd.Series(dict(vals))
for lab,aa in [('2020-22',x.loc['2020':'2022']),('2023-25',x.loc['2023':'2025']),('2026-27',x.loc['2026':'2027']),('2028YTD',x.loc['2028':])]: print(lab,len(aa),round(aa.mean(),6),round(aa.mean()/aa.std(ddof=1),6) if len(aa)>1 else np.nan)
print('coverage',round(f.notna().sum(axis=1).mean()/len(U),4),'turnover',round((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)>0.05).mean(),4))
# provenance artifact for deterministic checker
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20280713_residual_breadth_signal.csv',index=False)
