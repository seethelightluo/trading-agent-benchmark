import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s,macro=False):
 p=('../persistent/index_data/' if macro else '../persistent/stock_data/')+s+'.csv'
 d=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index(); return d.close
px=pd.concat({s:load(s) for s in U},axis=1,sort=False).sort_index().loc[:'2026-07-15']; r=px.pct_change()
m=pd.concat({'vix':load('VIX',1).pct_change(),'dxy':load('DXY',1).pct_change(),'us10':load('US10Y').pct_change()},axis=1,sort=False).reindex(px.index).ffill()
z=(m-m.rolling(60,min_periods=40).mean())/m.rolling(60,min_periods=40).std(); shock=(z['vix']+z['dxy']-z['us10'])/3
# explicit rolling beta, avoids pandas rolling alignment quirks
B=pd.DataFrame(index=px.index,columns=U,dtype=float)
for i in range(60,len(px)):
 x=shock.iloc[i-59:i+1].values
 for s in U:
  y=r[s].iloc[i-59:i+1].values; ok=np.isfinite(x)&np.isfinite(y)
  if ok.sum()>=40 and np.var(x[ok])>1e-12: B.iloc[i,B.columns.get_loc(s)]=np.cov(y[ok],x[ok],ddof=1)[0,1]/np.var(x[ok],ddof=1)
# candidate: stress beta, and stress-weighted cross-sectional residual of 20d returns
fac={'stress_beta':-B, 'stress_resid20':-(px.pct_change(20).sub(px.pct_change(20).median(axis=1),axis=0))*((1+shock.abs()).clip(upper=3).values[:,None])}
for name,f in fac.items():
 for h in [1,5,10]:
  vals=[]; ns=[]
  for i in range(len(px)-h):
   q=pd.concat([f.iloc[i],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
   if len(q)>=8:
    vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); ns.append(len(q))
  a=np.array(vals); print(name,h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
# regime and turnover for candidate
f=fac['stress_resid20']; q=f.rank(axis=1,pct=True); print('stress_resid20 turnover',q.diff().abs().mean().mean(),'coverage',f.notna().mean().mean())
for name,f in fac.items():
 zc=[]
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  for i in range(len(px)-1):
   if not (f.index[i].year>=lo and f.index[i].year<=hi): continue
   q=pd.concat([f.iloc[i],px.iloc[i+1]/px.iloc[i]-1],axis=1).dropna()
   if len(q)>=8: zc.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
  print(name,lo,hi,round(np.mean(zc[-100000:]),5),len(zc))
