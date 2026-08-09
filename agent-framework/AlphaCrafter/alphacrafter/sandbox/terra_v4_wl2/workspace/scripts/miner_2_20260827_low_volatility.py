import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try:
        D[s]=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date').loc[:'2026-07-15']
    except Exception as e: print('missing',s)
# Low-volatility factor: negative trailing realized volatility, with strictly completed close returns.
for w in [10,20,40,60]:
    rows=[]
    for s,x in D.items():
        r=x.close.pct_change(); f=-r.rolling(w,min_periods=max(8,w//2)).std()
        for i,dt in enumerate(x.index):
            if pd.notna(f.iloc[i]) and i+1<len(x): rows.append((dt,s,float(f.iloc[i]),float(x.close.iloc[i+1]/x.close.iloc[i]-1)))
    a=pd.DataFrame(rows,columns=['date','s','f','y']); ics=[]
    for dt,g in a.groupby('date'):
        if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: ics.append(spearmanr(g.f,g.y).statistic)
    z=np.array(ics)
    print('window',w,'dates',len(z),'avg_names',round(a.groupby('date').size().mean(),2),'coverage',round(a.s.nunique()/15,4),'IC',round(z.mean(),8),'ICIR',round(z.mean()/z.std(ddof=1),8),'hit',round(np.mean(z>0),4))
    for yr in range(2020,2027):
        q=[]
        for dt,g in a.groupby('date'):
            if dt.year==yr and len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:q.append(spearmanr(g.f,g.y).statistic)
        if q: print(' yr',yr,'IC',round(float(np.mean(q)),5),'ICIR',round(float(np.mean(q)/np.std(q,ddof=1)),4),'n',len(q))
