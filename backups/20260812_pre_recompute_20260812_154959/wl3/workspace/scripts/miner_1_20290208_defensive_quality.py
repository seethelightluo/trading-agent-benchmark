import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cs={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None: cs[s]=d.set_index('date').close
p=pd.concat(cs,axis=1).sort_index(); r=p.pct_change(); v=r.rolling(20).std(); down=r.clip(upper=0).abs().rolling(20).mean(); pos=r.gt(0).rolling(20).mean()
f=(1/v)*(pos/(1+down*10)); f=f.sub(f.median(axis=1),axis=0).shift(1); fw=p.shift(-10).div(p)-1
out=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
 if len(z)>=8: out.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
x=pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
for nm,q in [('all',x),('2026-27',x.loc['2026':'2027']),('2028+',x.loc['2028':]),('recent252',x.tail(252))]:
 a=q.ic; print(nm,len(a),round(q.n.mean(),2),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6),round((a>0).mean(),4))
print('coverage',f.notna().sum().sum()/(len(f)*15),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean()); f.to_csv('scripts/miner_1_20290208_defensive_quality_signal.csv')
