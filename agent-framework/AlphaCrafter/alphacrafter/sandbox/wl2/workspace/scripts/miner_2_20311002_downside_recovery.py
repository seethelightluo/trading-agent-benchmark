import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; idx='../persistent/index_data'
D={}
for s in U:
 p=f'{base}/{s}.csv'; q=f'{idx}/{s}.csv'
 try:d=pd.read_csv(p)
 except:d=pd.read_csv(q)
 d.date=pd.to_datetime(d.date); D[s]=d.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().ffill(); ret=px.pct_change()
down=ret.where(ret<0,0).rolling(30).std(); raw=(px/px.shift(30)-1)/(down*np.sqrt(252)+1e-8)
recovery=px/px.rolling(60).min()-1; fac=raw*(1+0.5*np.tanh(recovery/0.10))
for h in [1,3,5,10]:
 vals=[];ns=[]
 for dt in fac.index:
  a=fac.shift(1).loc[dt]; b=(px.shift(-h)/px-1).loc[dt];ok=a.notna()&b.notna()
  if ok.sum()>=8: vals.append(a[ok].corr(b[ok],method='spearman'));ns.append(ok.sum())
 x=pd.Series(vals).dropna();print('h',h,'dates',len(x),'avg_n',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(),6),'hit',round((x>0).mean(),4))
z=[]
for dt in fac.index:
 a=fac.shift(1).loc[dt];b=ret.shift(-1).loc[dt];ok=a.notna()&b.notna()
 if ok.sum()>=8:z.append((dt,a[ok].corr(b[ok],method='spearman'),ok.sum()))
z=pd.DataFrame(z,columns=['date','ic','n']).set_index('date')
for lo,hi in [('2020','2022'),('2023','2025'),('2026','2031')]:
 q=z.loc[lo:hi,'ic'];print('regime',lo,hi,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6))
print('coverage',round(fac.notna().stack().mean(),4));r=fac.rank(axis=1,pct=True);print('turnover',round(r.diff().abs().mean(axis=1).mean()/2,4))
out=fac.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20311002_downside_recovery_signal.csv',index=False)
