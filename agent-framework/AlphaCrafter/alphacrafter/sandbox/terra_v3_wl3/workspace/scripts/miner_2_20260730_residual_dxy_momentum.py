import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
root='../persistent/stock_data'; macro='../persistent/index_data'
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for r in [root,macro]:
  f=f'{r}/{s}.csv'
  if os.path.exists(f):
   d=pd.read_csv(f); d.date=pd.to_datetime(d.date); return d.set_index('date').close.sort_index()
 return None
px={s:load(s) for s in syms}; dxy=load('DXY')
px={s:v for s,v in px.items() if v is not None}
rets=pd.DataFrame({s:v.pct_change() for s,v in px.items()}).join(dxy.pct_change().rename('DXY'),how='inner')
ics={h:[] for h in [1,5,10]}; vals=[]
for i in range(70,len(rets)-10):
 dt=rets.index[i]; hist=rets.iloc[i-60:i]
 for s in px:
  x=hist.DXY; y=hist[s]; ok=x.notna()&y.notna()
  if ok.sum()<45 or np.var(x[ok],ddof=1)<1e-12: continue
  beta=np.cov(y[ok],x[ok],ddof=1)[0,1]/np.var(x[ok],ddof=1)
  fac=(1+(hist[s]-beta*hist.DXY).iloc[-20:].fillna(0)).prod()-1
  vals.append((dt,s,fac))
  for h in ics: ics[h].append((dt,s,fac,(1+rets[s].iloc[i+1:i+1+h].fillna(0)).prod()-1))
for h,x in ics.items():
 df=pd.DataFrame(x,columns=['date','s','f','r']); by=df.groupby('date').filter(lambda z:len(z)>=8).groupby('date').apply(lambda z:spearmanr(z.f,z.r).statistic)
 print(h,'dates',len(by),'avgN',df.groupby('date').size().mean(),'IC',round(by.mean(),5),'ICIR',round(by.mean()/by.std(ddof=1),5),'hit',round((by>0).mean(),4))
v=pd.DataFrame(vals,columns=['date','s','f']).pivot(index='date',columns='s',values='f'); rank=v.rank(pct=True)
print('coverage',round(v.notna().mean().mean(),4),'turnover',round((rank-rank.shift()).abs().mean().mean(),4),'span',v.index.min(),v.index.max())
reg=pd.DataFrame({s:px[s].pct_change(20) for s in px}); rev=-pd.DataFrame({s:px[s].pct_change(5) for s in px}); a=v.stack()
print('corr regular momentum',round(a.corr(reg.reindex(v.index).stack()),4),'corr reversal',round(a.corr(rev.reindex(v.index).stack()),4))
