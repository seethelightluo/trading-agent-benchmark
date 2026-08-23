import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
data={}
for s in U:
 d=get_stock_daily_data(s,days=2700)
 if d is not None and len(d)>120:
  d=d.copy(); d.date=pd.to_datetime(d.date); data[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(data).sort_index().ffill(); r=p.pct_change()
# Observation-only VIX: use only same-date close, and lag one session for decision availability.
v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); v=v.set_index('date').close.astype(float).reindex(p.index).ffill()
vr=v.pct_change(); vlevel=v.rolling(60).rank(pct=True).shift(1)
# Trend agreement, with high-volatility regime damping trend and emphasizing reversal.
m20=p.pct_change(20); m5=p.pct_change(5); m60=p.pct_change(60); vol=r.rolling(20).std()*np.sqrt(252)
ag=(np.sign(m5)+np.sign(m20)+np.sign(m60))/3
# regime coefficient is +1 normal, -0.5 when VIX is extreme; interpretable conditional trend
coef=pd.Series(1.0,index=p.index); coef[vlevel>0.8]=-0.5
f=(m20/vol)*ag.multiply(coef,axis=0); f=f.sub(f.mean(axis=1),axis=0)
def calc(h):
 fr=p.shift(-h).div(p).sub(1); vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): vals.append(c); ns.append(len(z))
 a=np.array(vals); icir=a.mean()/(a.std(ddof=1)/np.sqrt(len(a)))
 turn=(f.diff().abs().sum(axis=1)/(f.abs().sum(axis=1)*2)).replace([np.inf,-np.inf],np.nan).mean()
 return len(a),a.mean(),icir,np.mean(a>0),turn,np.mean(ns)
print('universe',len(data),'dates',len(p),'range',p.index.min(),p.index.max())
for h in [1,5,10,20]: print('H',h,'n IC ICIR hit turnover avgN',calc(h))
fr=p.shift(-10).div(p).sub(1); vals=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): vals.append((dt,c))
q=pd.DataFrame(vals,columns=['date','ic']); q['year']=q.date.dt.year
print(q.groupby('year').ic.agg(['count','mean']).tail(11).to_string())
print('coverage',f.notna().sum(axis=1).mean()/len(U),'vix dates',v.notna().sum())
