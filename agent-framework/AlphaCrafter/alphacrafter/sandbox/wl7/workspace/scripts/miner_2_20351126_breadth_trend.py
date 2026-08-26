import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def L(s):
 try:x=get_index_daily_data(s,5000)
 except:x=None
 if x is None:
  try:x=get_stock_daily_data(s,5000)
  except:x=None
 if x is None:return None
 return x.assign(date=pd.to_datetime(x['date'])).set_index('date').close.astype(float)
P={s:L(s) for s in U}; px=pd.DataFrame({s:x for s,x in P.items() if x is not None}).sort_index().ffill(limit=3); r=np.log(px).diff()
# broad, risk-adjusted medium-term trend, confidence from fraction of positive lagged asset returns
mom=.6*r.rolling(60).sum()+.4*r.rolling(20).sum(); dv=r.where(r<0,0).rolling(40).std().clip(lower=1e-5); breadth=(r>0).rolling(20).mean().mean(axis=1)
g=(breadth-breadth.rolling(120).mean())/breadth.rolling(120).std().clip(lower=1e-6); gate=(1/(1+np.exp(-g))).clip(.2,.8)
f=(mom/dv).mul(gate,axis=0).shift(1); fr=px.shift(-20)/px-1; z=[]
for d in f.index:
 a=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(a)>=8:z.append((d,a.iloc[:,0].corr(a.iloc[:,1],method='spearman'),len(a)))
x=pd.DataFrame(z,columns=['date','ic','n']).set_index('date').dropna(); m=x.ic.mean(); sd=x.ic.std(ddof=1)
print('factor=breadth_gated_multihorizon_trend_downsidevol_h20');print('dates',len(x),'avg_n',x.n.mean(),'coverage',f.notna().sum().sum()/f.size,'IC',m,'ICIR',m/sd*np.sqrt(252),'hit',(x.ic>0).mean());print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for a,b in [('2020','2024'),('2025','2029'),('2030','2034'),('2035','2035')]:
 y=x.loc[a:b].ic;print(a,len(y),y.mean(),y.mean()/y.std(ddof=1)*np.sqrt(252) if len(y)>2 else np.nan)
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20351126_breadth_trend_signal.csv',index=False)
