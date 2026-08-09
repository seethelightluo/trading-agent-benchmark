import pandas as pd, numpy as np
from scipy.stats import spearmanr
root='../persistent'; syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s,macro=False):
 p=f'{root}/index_data/{s}.csv' if macro else f'{root}/stock_data/{s}.csv'; d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); return pd.to_numeric(d['close'],errors='coerce')
px=pd.concat([load(s).rename(s) for s in syms],axis=1).loc[:'2026-07-15']; r=px.pct_change(fill_method=None)
dxy=load('DXY',True).reindex(px.index).ffill(); dr=dxy.pct_change(fill_method=None)
betas=pd.DataFrame(index=px.index,columns=syms,dtype=float)
for s in syms:
 q=pd.concat([r[s],dr],axis=1).dropna(); q.columns=['a','m']; b=q['a'].rolling(60,min_periods=45).cov(q['m'])/q['m'].rolling(60,min_periods=45).var(); betas.loc[b.index,s]=b
f=px.pct_change(20)-betas.mul(dxy.pct_change(20),axis=0); fr=r.shift(-1); ics=[];dates=[];ns=[]
for dt in f.index:
 a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(a)>=8:
  z=spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic
  if np.isfinite(z): ics.append(z);dates.append(dt);ns.append(len(a))
ic=pd.Series(ics,index=pd.DatetimeIndex(dates)); print('dates',len(ic),'avg_n',np.mean(ns),'coverage',len(ic)/len(f.index)); print('IC %.8f ICIR %.8f hit %.4f'%(ic.mean(),ic.mean()/ic.std(),(ic>0).mean()))
for y,g in ic.groupby(ic.index.year): print(y,len(g),round(g.mean(),6),round(g.mean()/g.std(),5))
for h in [5,10]:
 fh=px.pct_change(h).shift(-h); z=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fh.loc[dt]],axis=1).dropna()
  if len(a)>=8:z.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic)
 z=pd.Series(z).dropna();print('horizon',h,'IC',z.mean(),'ICIR',z.mean()/z.std(),'n',len(z))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_2_20260827_macro_neutral_mom.csv',index=False)
