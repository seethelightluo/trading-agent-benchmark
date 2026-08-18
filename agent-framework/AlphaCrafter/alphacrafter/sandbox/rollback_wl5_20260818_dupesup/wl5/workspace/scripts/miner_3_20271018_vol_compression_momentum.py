import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try: D[s]=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date')
    except Exception as e: print('missing',s,e)
# Candidate: volatility-compression breakout continuation. Low recent realized vol relative to long vol,
# conditioned by signed 10d momentum; use only information through t.
def fac(x):
    r=x.close.pct_change()
    short=r.rolling(5).std(); long=r.rolling(20).std()
    return x.close.pct_change(10)/(long+1e-12) * (short/(long+1e-12))
rec=[]
for s,x in D.items():
    f=fac(x)
    for i in range(len(x)-1):
        if x.index[i] >= pd.Timestamp('2020-02-01') and pd.notna(f.iloc[i]):
            rec.append((x.index[i],s,float(f.iloc[i]),float(x.close.iloc[i+1]/x.close.iloc[i]-1)))
a=pd.DataFrame(rec,columns=['date','symbol','factor','forward'])
for label,lo,hi in [('full','2020-02-01','2027-10-18'),('recent','2026-07-16','2027-10-18'),('early','2020-02-01','2023-12-31'),('mid','2024-01-01','2026-07-15')]:
    q=a[(a.date>=lo)&(a.date<=hi)]; ics=[]
    for d,g in q.groupby('date'):
        if len(g)>=8 and g.factor.nunique()>1 and g.forward.nunique()>1: ics.append(spearmanr(g.factor,g.forward).statistic)
    z=np.asarray(ics,float)
    print(label,'dates',len(z),'instruments',q.symbol.nunique(),'meanN',round(q.groupby('date').size().mean(),2),'coverage',round(q.symbol.nunique()/15,3),'IC',round(np.nanmean(z),6),'ICIR',round(np.nanmean(z)/np.nanstd(z,ddof=1),6),'hit',round(np.mean(z>0),4))
# 5d decay
for h in [1,3,5]:
    rec2=[]
    for s,x in D.items():
        f=fac(x)
        y=x.close.shift(-h)/x.close-1
        for i in range(len(x)):
            if pd.notna(f.iloc[i]) and pd.notna(y.iloc[i]) and x.index[i]>=pd.Timestamp('2020-02-01'):
                rec2.append((x.index[i],float(f.iloc[i]),float(y.iloc[i])))
    q=pd.DataFrame(rec2,columns=['date','f','y']); z=[]
    for d,g in q.groupby('date'):
        if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:z.append(spearmanr(g.f,g.y).statistic)
    z=np.asarray(z); print('decay',h,'IC',round(np.nanmean(z),6),'ICIR',round(np.nanmean(z)/np.nanstd(z,ddof=1),6),'dates',len(z))
