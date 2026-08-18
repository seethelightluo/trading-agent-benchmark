import numpy as np, pandas as pd, os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
    fn='../persistent/stock_data/'+s+'.csv'
    if os.path.exists(fn):
        d=pd.read_csv(fn)
        datecol='date' if 'date' in d else d.columns[0]
        close='close' if 'close' in d else 'Close'
        px[s]=pd.Series(d[close].values,index=pd.to_datetime(d[datecol]))
p=pd.DataFrame(px).sort_index().ffill()
r=p.pct_change()
up=r.where(r>0,0.0).rolling(30,min_periods=20).std()
dn=(-r.where(r<0,0.0)).rolling(30,min_periods=20).std()
base=(up/(dn+1e-8)).replace([np.inf,-np.inf],np.nan)
f=np.log(base.clip(0.1,10)).shift(1); f=f.sub(f.mean(axis=1),axis=0)
fwd=p.shift(-10)/p-1
rows=[]; sig=[]
for dt in f.index:
 x=f.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  ic=x[ok].corr(y[ok],method='spearman'); rows.append((dt,ic,ok.sum()))
  for s in U:
   if ok.get(s,False): sig.append((dt,s,float(x[s])))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').loc['2020-01-01':'2033-10-14']
mean=z.ic.mean(); sd=z.ic.std(ddof=1); icir=mean/sd*np.sqrt(252) if sd else np.nan
print('dates',len(z),'avgN',z.n.mean(),'coverage',len(sig)/(len(z)*len(U)))
print('IC10d',mean,'ICIR_daily_annualized',icir,'hit', (z.ic>0).mean(),'turnover', f.rank(axis=1,pct=True).diff().abs().mean().mean())
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2033')]:
 q=z.loc[a:b].ic
 if len(q): print(a,b,len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252) if len(q)>1 else np.nan)
pd.DataFrame(sig,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20331014_asymmetry_ratio_signal.csv',index=False)
z.to_csv('scripts/miner_2_20331014_asymmetry_ratio_ic.csv')
