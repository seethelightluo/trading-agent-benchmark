import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
    for fn in (get_index_daily_data,get_stock_daily_data):
        try:
            d=fn(s,5000)
            if d is not None and len(d)>=100:return d
        except Exception: pass
D={s:get(s) for s in U};D={s:d for s,d in D.items() if d is not None}
C=pd.DataFrame({s:d.set_index(pd.to_datetime(d.date)).close.astype(float) for s,d in D.items()}).sort_index()
R=C.pct_change(); r5=C.pct_change(5)
# Stress = broad negative tape OR unusually high cross-sectional dispersion.
breadth=(r5<0).mean(axis=1); disp=r5.std(axis=1)
stress=(breadth>=0.60)|(disp>=disp.rolling(120,min_periods=60).quantile(.80))
med=r5.median(axis=1); res=r5.sub(med,axis=0)
vol=R.rolling(20).std()*np.sqrt(5)
f=(-res/vol.replace(0,np.nan)).clip(-3,3)
f=f.where(stress, np.nan); f=f.sub(f.mean(axis=1),axis=0)
fr=C.pct_change().shift(-1)
rows=[]
for d in f.index:
 q=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: rows.append((d,q.iloc[:,0].rank().corr(q.iloc[:,1].rank()),len(q)))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('assets',len(D),'price_dates',len(C),'stress_days',int(stress.sum()),'IC_dates',len(o),'avg_n',round(o.n.mean(),3),'coverage_conditional',round(o.n.mean()/15,4))
print('IC %.6f ICIR %.6f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(),(o.ic>0).mean()))
for a,b in [('2026','2029'),('2030','2032')]:
 q=o.loc[a:b].ic;print(a,b,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(),'hit',(q>0).mean() if len(q) else np.nan)
for h in [3,5,10]:
 rr=C.pct_change(h).shift(-h); z=[]
 for d in f.index:
  q=pd.concat([f.loc[d],rr.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:z.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 print('decay',h,'IC',np.nanmean(z),'dates',len(z))
print('recent120',o.tail(120).ic.mean(),o.tail(120).ic.mean()/o.tail(120).ic.std(),'dates',len(o.tail(120)))
# signal artifact for provenance
f.to_csv('scripts/miner_3_20320722_stress_residual_reversal_signal.csv',index_label='date')
