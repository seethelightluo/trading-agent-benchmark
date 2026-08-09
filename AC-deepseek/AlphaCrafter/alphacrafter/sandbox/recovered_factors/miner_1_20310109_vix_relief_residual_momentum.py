import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
def load(sym):
 fs=glob.glob(base+'/'+sym+'.csv')+glob.glob(base+'/'+sym+'/*.csv')
 if not fs: fs=glob.glob('../persistent/index_data/'+sym+'.csv')
 d=pd.read_csv(fs[0]); d['date']=pd.to_datetime(d['date']); return d.set_index('date')['close'].astype(float)
px=pd.concat({a:load(a) for a in assets},axis=1).sort_index()
r=np.log(px).diff()
def macro(sym):
 d=pd.read_csv('../persistent/index_data/'+sym+'.csv'); d['date']=pd.to_datetime(d.date); return d.set_index('date').close.astype(float)
v=np.log(macro('VIX')).diff().reindex(px.index); dx=np.log(macro('DXY')).diff().reindex(px.index)
# relief transition: both macro risk gauges falling over recent 5 days
mask=(v.rolling(5).sum()<0)&(dx.rolling(5).sum()<0)
# factor: residual medium-term momentum, activated only in relief; demean each date
mom=r.rolling(20).sum(); fac=mom.sub(mom.mean(axis=1),axis=0).where(mask, np.nan)
rows=[]
for i in range(len(px)-21):
 if not mask.iloc[i]: continue
 x=fac.iloc[i]; y=r.iloc[i+1:i+11].sum()
 z=pd.concat([x,y],axis=1).dropna();
 if len(z)>=8: rows.append((px.index[i],len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
q=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('candidate=relief_residual_mom20; dates',len(q),'meanN',q.n.mean(),'coverage cells',fac.notna().sum().sum(),'total',fac.size)
for h in [1,5,10,20]:
 rr=[]
 for i in range(len(px)-h):
  if not mask.iloc[i]: continue
  z=pd.concat([fac.iloc[i],r.iloc[i+1:i+1+h].sum()],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 rr=np.array(rr); print('H',h,'IC',rr.mean(),'ICIR',rr.mean()/rr.std(ddof=1),'hit',np.mean(rr>0),'n',len(rr))
for period in [(2020,2023),(2024,2027),(2028,2030)]:
 a=q[(q.index.year>=period[0])&(q.index.year<=period[1])].ic
 print('regime',period,'dates',len(a),'IC',a.mean() if len(a) else np.nan,'ICIR',a.mean()/a.std(ddof=1) if len(a)>1 else np.nan)
# turnover sampled 10d active ranks
print('valid asset coverage',fac.notna().sum().sum()/fac.size)
