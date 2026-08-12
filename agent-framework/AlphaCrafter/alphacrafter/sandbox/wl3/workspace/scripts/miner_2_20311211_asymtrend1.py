import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  x=pd.read_csv(f); x['date']=pd.to_datetime(x['date']); x=x[x.date<=pd.Timestamp('2031-12-11')].sort_values('date').drop_duplicates('date'); D[s]=x.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); lp=np.log(p); r=lp.diff(); r20=lp-lp.shift(20)
up=(r>0).rolling(20,min_periods=15).mean(); dn=r.where(r<0).rolling(40,min_periods=20).std(); allv=r.rolling(40,min_periods=20).std()
raw=(r20*(0.5+up))/(dn*np.sqrt(40)+0.5*allv*np.sqrt(40)+1e-8); f=raw.sub(raw.median(axis=1),axis=0).shift(1); fr=lp.shift(-1)-lp
rows=[]
for dt in f.index:
 a,b=f.loc[dt],fr.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8 and a[ok].nunique()>1: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic
print('shape',p.shape,'valid_dates',len(z),'assets',len(D),'avgN',z.n.mean(),'coverage',z.n.mean()/len(D))
print('H1 IC %.8f ICIR %.8f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
for lo,hi in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030'),('2031','2031')]:
 x=q.loc[lo:hi]; print(lo,len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1) if len(x)>2 else np.nan)
for n in [60,120,252]:
 x=q.tail(n); print('recent',n,len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1) if len(x)>2 else np.nan)
print('turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean()); f.to_csv('scripts/miner_2_20311211_asymtrend_signal.csv'); z.to_csv('scripts/miner_2_20311211_asymtrend_ic.csv')
