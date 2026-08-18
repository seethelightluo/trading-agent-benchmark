import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>=120:
  d=d.copy(); d['date']=pd.to_datetime(d['date']); D[s]=d.set_index('date')['close'].astype(float).sort_index()
p=pd.DataFrame(D).sort_index(); r=np.log(p).diff(); mom=p/p.shift(20)-1
sv=r.rolling(10).std(); lv=r.rolling(40).std(); comp=-(sv/lv-1)
disp=r.std(axis=1).rolling(20).mean(); stress=(disp/disp.rolling(120).median()).clip(.5,2); gate=(2-stress).clip(0,1.5)
f=(mom.rank(axis=1,pct=True)-.5)*(comp.rank(axis=1,pct=True)-.5)*gate.values[:,None]; f=f.shift(1); fr=p.shift(-10)/p-1
ics=[]; ns=[]; turns=[]; prev=None
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 q=f.loc[dt].rank(pct=True)
 if prev is not None: turns.append((q-prev).abs().mean())
 prev=q
ic=np.asarray(ics); ic=ic[np.isfinite(ic)]
print({'factor':'stress_conditioned_compression_momentum','dates':len(ic),'avg_n':float(np.mean(ns)),'min_n':int(np.min(ns)),'ic':float(np.mean(ic)),'icir':float(np.mean(ic)/(np.std(ic,ddof=1)/np.sqrt(len(ic)))),'hit':float(np.mean(ic>0)),'turnover':float(np.mean(turns)),'coverage':float(np.mean(ns)/15)})
for label,a in [('early',ic[:len(ic)//3]),('middle',ic[len(ic)//3:2*len(ic)//3]),('late',ic[2*len(ic)//3:]),('recent250',ic[-250:])]: print(label,float(np.mean(a)),float(np.mean(a)/(np.std(a,ddof=1)/np.sqrt(len(a)))))
print('cutoff',str(p.index.max().date()),'assets',len(D))
