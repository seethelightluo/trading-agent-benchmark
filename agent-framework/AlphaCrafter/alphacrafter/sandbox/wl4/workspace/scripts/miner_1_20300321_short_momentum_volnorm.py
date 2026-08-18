import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try: x=get_stock_daily_data(s, days=2600)
    except Exception as e: print('skip',s,str(e)); continue
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); D[s]=x.set_index('date')['close'].astype(float).sort_index()
p=pd.concat(D,axis=1).sort_index().ffill(); r=np.log(p).diff()
f=(np.log(p).diff(5)/r.rolling(20).std()).shift(1); fr=np.log(p).shift(-10)-np.log(p)
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
z=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
for label,q in [('full',z),('recent250',z.tail(250)),('early',z.iloc[:len(z)//3]),('middle',z.iloc[len(z)//3:2*len(z)//3]),('late',z.iloc[2*len(z)//3:])]:
 ic=q.ic.mean(); sd=q.ic.std(ddof=1); print(label,'dates',len(q),'avgN',round(q.n.mean(),2),'minN',q.n.min(),'IC',round(ic,6),'ICIR',round(ic/sd*np.sqrt(252),6),'hit',round((q.ic>0).mean(),4))
rank=f.rank(axis=1,pct=True); print('coverage',round(f.notna().sum().sum()/(f.shape[0]*f.shape[1]),6),'turnover',round(rank.diff().abs().mean(axis=1).dropna().mean(),6),'assets',len(D),'rows',len(p))
for h in [5,20]:
 ff=np.log(p).shift(-h)-np.log(p); rr=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(q)>=8: rr.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 print('horizon',h,'IC',round(np.nanmean(rr),6),'n',len(rr),'ICIR',round(np.nanmean(rr)/np.nanstd(rr,ddof=1)*np.sqrt(252),6))
