import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
cl={}
for a in assets:
    d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date')
    d=d[d.date<=cut].set_index('date')
    cl[a]=d.close
px=pd.DataFrame(cl).sort_index()
for look in [2,3,10,20]:
    fac=-px.pct_change(look,fill_method=None)
    fac.to_csv(f'scripts/miner_2_20270325_plain_reversal_{look}d_signal.csv')
    vals=[]; dates=[]; ns=[]
    fwd=px.pct_change(1,fill_method=None).shift(-1)
    for dt in fac.index:
        z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
        if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
            vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); ns.append(len(z))
    s=pd.Series(vals,index=dates)
    print('look',look,'dates',len(s),'avgN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'coverage',round(fac.notna().sum(axis=1).mean()/15,4),'turn',round(fac.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
    for label,lo,hi in [('early','2020-01-01','2022-12-31'),('mid','2023-01-01','2025-12-31'),('late','2026-01-01','2027-03-24')]:
        q=s[(s.index>=lo)&(s.index<=hi)]
        print(label,'n',len(q),'IC',round(q.mean(),6) if len(q) else None,'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)
