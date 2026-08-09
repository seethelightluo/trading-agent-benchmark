import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in syms:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date')
 # point-in-time signal at t; next close return
 r=d.close.pct_change()
 vol=np.log(d.volume.replace(0,np.nan))
 surprise=(vol-vol.rolling(20,min_periods=10).median())/(vol.rolling(20,min_periods=10).std()+1e-12)
 # volume-confirmed reversal: reversal is stronger after unusually high volume
 sig=-r.rolling(3,min_periods=3).sum()*np.clip(surprise, -2, 3)
 fwd=d.close.shift(-1)/d.close-1
 z=pd.DataFrame({'date':d.date,'asset':s,'signal':sig,'fwd':fwd})
 rows.append(z)
x=pd.concat(rows).dropna()
ics=[]; dates=[]
for dt,g in x.groupby('date'):
 if len(g)>=8 and g.signal.nunique()>1 and g.fwd.nunique()>1:
  ics.append(g.signal.corr(g.fwd,method='spearman'));dates.append(dt)
a=np.array(ics);print('dates',len(a),'assets/date mean',x.groupby('date').size().mean(),'coverage',len(x)/(len(pd.concat(rows))*1.0))
print('IC',np.nanmean(a),'ICIR',np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12),'hit',np.mean(a>0),'median',np.nanmedian(a))
# pearson too
pp=[]
for dt,g in x.groupby('date'):
 if len(g)>=8: pp.append(g.signal.corr(g.fwd))
print('pearson',np.nanmean(pp))
# horizons
for h in [1,3,5]:
 z=[]
 for s in syms:
  d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date');r=d.close.pct_change();v=np.log(d.volume.replace(0,np.nan)); ss=(v-v.rolling(20,min_periods=10).median())/(v.rolling(20,min_periods=10).std()+1e-12); sig=-r.rolling(3,min_periods=3).sum()*np.clip(ss,-2,3); fw=d.close.shift(-h)/d.close-1;z.append(pd.DataFrame({'date':d.date,'signal':sig,'fwd':fw}))
 q=pd.concat(z).dropna();ii=[g.signal.corr(g.fwd,method='spearman') for _,g in q.groupby('date') if len(g)>=8];print('h',h,'n',len(ii),'IC',np.nanmean(ii),'IR',np.nanmean(ii)/(np.nanstd(ii,ddof=1)+1e-12))
# rank turnover
ranks=x.assign(rank=x.groupby('date').signal.rank(pct=True)); piv=ranks.pivot(index='date',columns='asset',values='rank');print('turnover',piv.diff().abs().mean().mean())
# save artifact
out=x[['date','asset','signal']].sort_values(['date','asset']);out.to_csv('../persistent/factor_signals_miner_3_20270225_volume_confirmed_reversal.csv',index=False)
print('artifact',len(out))
