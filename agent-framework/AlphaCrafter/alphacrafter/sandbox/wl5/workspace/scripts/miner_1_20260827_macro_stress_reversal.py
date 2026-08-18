import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# macro from persistent csv
v=pd.read_csv('../persistent/index_data/VIX.csv')
v['date']=pd.to_datetime(v['date']); v=v.set_index('date')['close'].sort_index()
px={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is not None and len(d):
  d=d.copy(); d['date']=pd.to_datetime(d['date']); px[s]=d.set_index('date')['close'].sort_index()
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# macro stress-adjusted short reversal, causal: VIX 10d return known at t, forward t+1
vix=v.reindex(P.index).ffill(); vr=vix.pct_change(10).clip(-1,1)
f=-r.rolling(5).sum().mul(1+vr,axis=0)
fr=P.pct_change().shift(-1)
ics=[]; dates=[]; ranks=[]
for dt in f.index:
 x=f.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt)
  ranks.append(x.reindex(z.index).rank(pct=True))
ic=pd.Series(ics,index=pd.to_datetime(dates)).dropna()
print('N dates',len(ic),'N instruments avg',round(np.mean([len(pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()) for d in dates]),2),'coverage',round(f.notna().mean().mean(),4))
print('daily IC %.6f ICIR %.6f hit %.4f'%(ic.mean(),ic.mean()/ic.std(),(ic>0).mean()))
for yr,g in ic.groupby(ic.index.year): print(yr,len(g),round(g.mean(),5),round(g.mean()/g.std(),4))
# turnover top/bottom rank changes
R=pd.DataFrame(ranks,index=pd.to_datetime(dates)); print('rank turnover',R.diff().abs().mean().mean())
# correlations with existing formulas approximate
for name,ff in [('rev',-r.rolling(5).sum()),('mom',r.rolling(20).sum()),('clv',(P-P.rolling(1).min())/(P.rolling(1).max()-P.rolling(1).min()))]:
 a=pd.concat([f.stack(),ff.stack()],axis=1).dropna(); print('corr',name,a.corr(method='spearman').iloc[0,1])
