import pandas as pd, numpy as np
from pathlib import Path
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 f=Path('../persistent/stock_data')/(s+'.csv')
 if not f.exists(): f=Path('../persistent/index_data')/(s+'.csv')
 d=pd.read_csv(f,parse_dates=['date']).drop_duplicates('date').set_index('date')
 return d['close'].astype(float)
p=pd.concat({s:load(s) for s in syms},axis=1).sort_index().loc[:'2026-12-02'].ffill()
r=p.pct_change(fill_method=None)
short=r.rolling(5,min_periods=4).std(); long=r.rolling(40,min_periods=20).std()
f=(-r.rolling(5,min_periods=5).sum()*(short/long).clip(0.5,3.0)).replace([np.inf,-np.inf],np.nan)
rows=[]; sig=[]
for h in [1,5,10]:
 fr=p.shift(-h)/p-1; ics=[]
 for dt in f.index:
  x=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(x)>=8: ics.append(x.iloc[:,0].corr(x.iloc[:,1]))
 a=np.array(ics); rows.append((h,len(a),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0)))
for dt in f.index:
 if f.loc[dt].notna().sum()>=8: sig.append([dt.strftime('%Y-%m-%d')]+[float(f.loc[dt,s]) if pd.notna(f.loc[dt,s]) else '' for s in syms])
print('dates',len(p),'avg instruments',f.notna().sum(axis=1).mean(),'coverage',f.notna().sum().sum()/(len(f)*15))
print('metrics h,n,IC,ICIR,hit'); [print(x) for x in rows]
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
pd.DataFrame(sig,columns=['date']+syms).to_csv('scripts/miner_2_20261203_volshock_reversal_signal.csv',index=False)
