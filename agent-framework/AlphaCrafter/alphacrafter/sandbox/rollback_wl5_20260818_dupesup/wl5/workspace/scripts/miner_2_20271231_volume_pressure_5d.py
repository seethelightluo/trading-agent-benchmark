import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-12-31'); D={}
for s in U:
    try:
        x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date')
        D[s]=x.loc[:END]
    except Exception as e: print('skip',s,e)

def factor(x):
    r=x.close.pct_change()
    v=x.volume.replace(0,np.nan)
    return (r*v).rolling(5,min_periods=4).sum()/(v.rolling(5,min_periods=4).sum()+1e-12)
for h in [1,5,10]:
    rec=[]
    for s,x in D.items():
        f=factor(x)
        y=x.close.shift(-h)/x.close-1
        for d in x.index:
            if d<=END and pd.notna(f.loc[d]) and pd.notna(y.loc[d]): rec.append((d,s,f.loc[d],y.loc[d]))
    a=pd.DataFrame(rec,columns=['d','s','f','y']); z=[]; ns=[]
    for d,g in a.groupby('d'):
        if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:
            z.append(spearmanr(g.f,g.y).statistic); ns.append(len(g))
    z=np.array(z); print('volume_pressure_5d h',h,'dates',len(z),'avgN',np.mean(ns),'IC',np.mean(z),'ICIR',np.mean(z)/np.std(z,ddof=1),'hit',np.mean(z>0),'coverage',a.s.nunique()/15)
# regime split for 10d
h=10; rec=[]
for s,x in D.items():
 f=factor(x); y=x.close.shift(-h)/x.close-1
 for d in x.index:
  if pd.notna(f.loc[d]) and pd.notna(y.loc[d]):rec.append((d,s,f.loc[d],y.loc[d]))
a=pd.DataFrame(rec,columns=['d','s','f','y'])
for label,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-12-31'),('2027','2027-01-01','2027-12-31')]:
 z=[]
 for d,g in a[(a.d>=lo)&(a.d<=hi)].groupby('d'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:z.append(spearmanr(g.f,g.y).statistic)
 z=np.array(z);print(label,len(z),np.mean(z),np.mean(z)/np.std(z,ddof=1) if len(z)>1 else np.nan)
# save artifact 10d
out=a.copy(); out.to_csv('scripts/miner_2_20271231_volume_pressure_5d_signal.csv',index=False)
