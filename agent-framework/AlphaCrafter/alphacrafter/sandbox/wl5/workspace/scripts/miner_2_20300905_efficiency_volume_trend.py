import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'; x=pd.read_csv(f); x['date']=pd.to_datetime(x['date']); x=x.sort_values('date').set_index('date'); D[s]=x
close=pd.DataFrame({s:D[s]['close'] for s in U}); vol=pd.DataFrame({s:D[s]['volume'] for s in U})
# Novel interpretable factor: medium trend efficiency, confirmed by relative volume.
r=close.pct_change(); ret20=close/close.shift(20)-1
path=r.abs().rolling(20).sum(); efficiency=ret20/path.replace(0,np.nan)
volratio=vol.rolling(5).mean()/vol.rolling(60).mean().replace(0,np.nan)
# volume confirmation mildly scales signed trend; cross-sectional rank keeps comparability
factor=efficiency * volratio.clip(0.5,2.0)
factor=factor.replace([np.inf,-np.inf],np.nan)
fr=close.shift(-10)/close-1
rows=[]
for dt in factor.index:
 a=factor.loc[dt]; b=fr.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
x=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
# same-horizon daily paper IC and ICIR (mean/std); turnover rank changes
print('dates',len(x),'avg_n',x.n.mean(),'coverage',x.n.sum()/(len(x)*15))
print('ic',x.ic.mean(),'icir',x.ic.mean()/x.ic.std(ddof=1),'hit', (x.ic>0).mean())
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2030-08-21')]:
 q=x.loc[a:b]; print(a,b,'dates',len(q),'ic',q.ic.mean(),'icir',q.ic.mean()/q.ic.std(ddof=1) if len(q)>1 else np.nan)
# signal turnover using cross-sectional ranks
rank=factor.rank(axis=1,pct=True); turn=(rank.diff().abs().mean(axis=1)).dropna(); print('turnover',turn.mean())
# signal artifact for provenance
out=factor.copy(); out.index.name='date'; out.reset_index().to_csv('scripts/miner_2_20300905_efficiency_volume_trend_signal.csv',index=False)
