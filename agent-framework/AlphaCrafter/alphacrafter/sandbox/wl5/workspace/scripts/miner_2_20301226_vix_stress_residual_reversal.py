import pandas as pd, numpy as np
from pathlib import Path
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2030-12-25'); base=Path('../persistent/stock_data')
def load(a):
 d=pd.read_csv(base/(a+'.csv')); d.date=pd.to_datetime(d.date); return pd.to_numeric(d.set_index('date').sort_index().close,errors='coerce')
px=pd.concat([load(a).rename(a) for a in assets],axis=1).sort_index().loc[:cut]
vix=pd.read_csv('../persistent/index_data/VIX.csv'); vix.date=pd.to_datetime(vix.date); vix=pd.to_numeric(vix.set_index('date').sort_index().close,errors='coerce').reindex(px.index).ffill()
r20=px.pct_change(20); bench=r20.mean(axis=1); cov=r20.rolling(60,min_periods=40).cov(bench); var=bench.rolling(60,min_periods=40).var(); beta=cov.div(var,axis=0); res=r20.sub(beta.mul(bench,axis=0),axis=0); rv=res.rolling(60,min_periods=40).std(); vr=vix.shift(1).rolling(252,min_periods=100).rank(pct=True); f=(-res/rv).mul(.5+vr,axis=0).shift(1)
rows=[]
for dt in f.index:
 x=f.loc[dt]; y=px.shift(-10).div(px).sub(1).loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((dt,x[ok].corr(y[ok]),ok.sum()))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z=o.ic.dropna(); print('dates',len(z),'mean_n',o.n.mean(),'coverage',o.n.sum()/(len(z)*15),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean()); print('period',o.index.min().date(),o.index.max().date()); f.to_csv('scripts/miner_2_20301226_vix_stress_residual_reversal_signal.csv',index_label='date')
