import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for a in assets:
    f=os.path.join(base,a+'.csv')
    d=pd.read_csv(f)
    d['date']=pd.to_datetime(d['date'])
    d=d.sort_values('date').set_index('date')
    col='close' if 'close' in d else d.columns[0]
    px[a]=d[col].astype(float)
p=pd.DataFrame(px).sort_index()
r=p.pct_change()
# candidate: risk-adjusted relative strength, with stress-aware sign; all inputs lagged one day
ret20=p.pct_change(20)
vol20=r.rolling(20).std()
medret=ret20.median(axis=1)
# cross-asset relative strength residual, robust to common market moves
raw=ret20.sub(medret,axis=0)
# defensive preference: in broad weak regimes, amplify resilient assets; in strong regimes same ranking
fac=(raw/vol20).shift(1)
# forward one-day return from date t to t+1, observations date t
fwd=p.pct_change().shift(-1)
rows=[]
for dt in p.index:
    x=fac.loc[dt]; y=fwd.loc[dt]
    z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
        rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
o=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
for label, sub in [('all',o),('2020_22',o.loc['2020':'2022']),('2023_25',o.loc['2023':'2025']),('2026_27',o.loc['2026':'2027']),('2028',o.loc['2028':])]:
    ic=sub.ic.dropna(); print(label,'dates',len(ic),'avgN',round(sub.n.mean(),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6) if len(ic)>1 else np.nan,'hit',round((ic>0).mean(),4))
# horizons decay
for h in [1,3,5,10]:
    y=p.pct_change(h).shift(-h)
    rr=[]
    for dt in p.index:
      z=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
      if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
    q=pd.Series(rr).dropna(); print('h',h,'N',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
print('coverage',fac.notna().sum().sum()/(len(p)*len(assets)),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
