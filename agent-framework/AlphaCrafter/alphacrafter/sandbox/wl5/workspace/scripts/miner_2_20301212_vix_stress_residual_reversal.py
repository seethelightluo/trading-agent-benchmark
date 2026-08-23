import pandas as pd, numpy as np
from pathlib import Path
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-12-11'); base=Path('../persistent/stock_data')
def load(p):
 d=pd.read_csv(p); d['date']=pd.to_datetime(d['date']); d=d.set_index('date').sort_index(); return pd.to_numeric(d['close'],errors='coerce')
px=pd.concat([load(base/(a+'.csv')).rename(a) for a in assets],axis=1).sort_index().loc[:cut]
vix=load(Path('../persistent/index_data/VIX.csv')).reindex(px.index).ffill()
r20=px.pct_change(20); bench=r20.mean(axis=1)
cov=r20.rolling(60,min_periods=40).cov(bench); var=bench.rolling(60,min_periods=40).var(); beta=cov.div(var,axis=0)
res=r20.sub(beta.mul(bench,axis=0),axis=0); rv=res.rolling(60,min_periods=40).std(); raw=-res/rv
vrank=vix.shift(1).rolling(252,min_periods=100).rank(pct=True)
f=raw.mul(0.5+vrank,axis=0).shift(1); fr=px.shift(-10).div(px)-1
rows=[]
for dt in f.index:
 x,y=f.loc[dt],fr.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((dt,x[ok].corr(y[ok]),ok.sum()))
out=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
z=out.ic.dropna(); print('dates',len(z),'mean_n',out.n.mean(),'coverage',out.n.sum()/(len(z)*15),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
for h in [5,10,20]:
 yy=px.shift(-h).div(px)-1; rr=[]
 for dt in f.index:
  x,y=f.loc[dt],yy.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8: rr.append(x[ok].corr(y[ok]))
 q=pd.Series(rr).dropna(); print('horizon',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
f.to_csv('scripts/miner_2_20301212_vix_stress_residual_reversal_signal.csv',index_label='date')
print('period',out.index.min().date(),out.index.max().date())
