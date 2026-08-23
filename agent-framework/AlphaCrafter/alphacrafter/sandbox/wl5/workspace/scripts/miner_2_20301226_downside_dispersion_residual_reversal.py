import pandas as pd, numpy as np
from pathlib import Path
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-12-25'); base=Path('../persistent/stock_data')
def load(a,col='close'):
 d=pd.read_csv(base/(a+'.csv')); d.date=pd.to_datetime(d.date); d=d.set_index('date').sort_index(); return pd.to_numeric(d[col],errors='coerce')
px=pd.concat([load(a).rename(a) for a in assets],axis=1).loc[:cut]
r=px.pct_change(); bench=r.mean(axis=1)
cov=r.rolling(60,min_periods=40).cov(bench); var=bench.rolling(60,min_periods=40).var(); beta=cov.div(var,axis=0)
r20=px.pct_change(20); b20=r20.mean(axis=1)
res=r20.sub(beta.mul(b20,axis=0),axis=0)
# downside residual risk; high cross-sectional dispersion makes reversal more actionable
neg=res.clip(upper=0)
down=neg.pow(2).rolling(60,min_periods=40).mean().pow(.5)
raw=-res/down
csdisp=res.std(axis=1).rolling(60,min_periods=40).rank(pct=True)
f=raw.mul(0.5+csdisp,axis=0).shift(1)
rows=[]
for dt in f.index:
 y=px.shift(-10).div(px)-1; x=f.loc[dt]; yy=y.loc[dt]; ok=x.notna()&yy.notna()
 if ok.sum()>=8: rows.append((dt,x[ok].corr(yy[ok]),ok.sum()))
out=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z=out.ic.dropna()
print('dates',len(z),'mean_n',round(out.n.mean(),2),'coverage',round(out.n.sum()/(len(z)*15),4),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
for h in [1,5,10,20]:
 y=px.shift(-h).div(px)-1; q=[]
 for dt in f.index:
  x=f.loc[dt]; yy=y.loc[dt]; ok=x.notna()&yy.notna()
  if ok.sum()>=8:q.append(x[ok].corr(yy[ok]))
 q=pd.Series(q).dropna(); print('horizon',h,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
print('period',out.index.min().date(),out.index.max().date())
f.to_csv('scripts/miner_2_20301226_downside_dispersion_residual_reversal_signal.csv',index_label='date')
