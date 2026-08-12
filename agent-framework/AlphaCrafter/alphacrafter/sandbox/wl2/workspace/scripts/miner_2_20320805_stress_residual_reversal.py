import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    for fn in (get_index_daily_data,get_stock_daily_data):
        try:
            x=fn(s,5000)
            if x is not None and len(x)>=120:return x
        except Exception: pass
D={s:fetch(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
C=pd.DataFrame({s:x.set_index(pd.to_datetime(x.date)).close.astype(float) for s,x in D.items()}).sort_index()
R=C.pct_change(); r5=C.pct_change(5)
# Residual reversal, activated by broad weakness or unusually high cross-sectional dispersion.
breadth=(r5<0).mean(axis=1); disp=r5.std(axis=1)
stress=(breadth>=.60)|(disp>=disp.rolling(120,min_periods=60).quantile(.80))
res=r5.sub(r5.median(axis=1),axis=0)
vol=R.rolling(20).std()*np.sqrt(5)
f=(-res/vol.replace(0,np.nan)).clip(-3,3).where(stress,np.nan)
f=f.sub(f.mean(axis=1),axis=0)
def eval(h):
    fr=C.pct_change(h).shift(-h); rows=[]
    for d in f.index:
        q=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
        if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
            rows.append((d,q.iloc[:,0].rank().corr(q.iloc[:,1].rank()),len(q)))
    o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
    return o
O=eval(1)
print('assets',len(D),'price_dates',len(C),'stress_days',int(stress.sum()),'IC_dates',len(O),'avg_n',round(O.n.mean(),3),'coverage',round(O.n.mean()/15,4))
print('daily IC %.6f ICIR %.6f hit %.4f'%(O.ic.mean(),O.ic.mean()/O.ic.std(),(O.ic>0).mean()))
for a,b in [('2020','2022'),('2023','2025'),('2026','2029'),('2030','2032')]:
 q=O.loc[a:b].ic; print(a+'-'+b,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6) if len(q)>1 else np.nan)
for h in [3,5,10]:
 q=eval(h);print('decay',h,'IC',round(q.ic.mean(),6),'dates',len(q))
q=O.tail(120).ic;print('recent120',round(q.mean(),6),round(q.mean()/q.std(),6),'dates',len(q))
f.to_csv('scripts/miner_2_20320805_stress_residual_reversal_signal.csv',index_label='date')
