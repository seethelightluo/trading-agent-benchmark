import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def fetch(s):
    d=get_stock_daily_data(s, 4000)
    if d is None or len(d)==0: d=get_index_daily_data(s,4000)
    if d is None: return None
    d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.drop_duplicates('date').set_index('date').sort_index()
    return d['close'].astype(float)
px={s:fetch(s) for s in U}; px={s:x for s,x in px.items() if x is not None}
prices=pd.DataFrame(px).sort_index(); r=prices.pct_change()
# interpretable multi-horizon trend consistency, normalized by 20d realized vol
m20=prices.pct_change(20); m60=prices.pct_change(60); m120=prices.pct_change(120)
vol=r.rolling(20).std()*np.sqrt(20)
# equal-weighted signed horizon returns, with longer horizon stabilizing signal
f=((m20+m60+m120)/3).div(vol.replace(0,np.nan), axis=0)
# require agreement: damp mixed-sign horizons, retain direction and magnitude
agree=(np.sign(m20)+np.sign(m60)+np.sign(m120)).abs()/3
f=f*agree
rows=[]
for dt in f.index:
    y=r.shift(-1).rolling(10).sum().shift(-9).loc[dt] # t+1..t+10
    z=pd.concat([f.loc[dt],y],axis=1).dropna()
    if len(z)>=8:
        ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
        rows.append((dt,ic,len(z)))
out=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
# note rolling expression above is forward sum: explicitly verify by direct aligned target
future=(prices.shift(-10)/prices-1)
rows=[]
for dt in f.index:
    z=pd.concat([f.loc[dt],future.loc[dt]],axis=1).dropna()
    if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
out=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
mu=out.ic.mean(); sd=out.ic.std(ddof=1); icir=mu/sd*np.sqrt(252) if sd else np.nan
print('dates',len(out),'assets',len(prices.columns),'avg_n',out.n.mean(),'coverage',f.notna().sum(axis=1).mean()/len(U))
print('10d IC %.6f ICIR %.6f hit %.4f turnover %.4f'%(mu,icir,(out.ic>0).mean(),f.rank(axis=1,pct=True).diff().abs().mean().mean()))
for a,b in [('2020','2024'),('2025','2026'),('2027','2029'),('2030','2033')]:
 q=out.loc[a:b,'ic']; print(a,b,'n',len(q),'ic',q.mean(),'icir',q.mean()/q.std(ddof=1)*np.sqrt(252) if len(q)>1 else np.nan)
print('decay')
for h in [5,10,20,40]:
 fu=prices.shift(-h)/prices-1; rr=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fu.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print(h,len(rr),np.nanmean(rr),np.nanmean(rr)/np.nanstd(rr,ddof=1)*np.sqrt(252))
