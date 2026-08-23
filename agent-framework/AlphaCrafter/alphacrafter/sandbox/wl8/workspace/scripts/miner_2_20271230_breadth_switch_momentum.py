import numpy as np, pandas as pd, os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
    for root in ['../persistent/index_data/','../persistent/stock_data/']:
        f=root+s+'.csv'
        if os.path.exists(f):
            d=pd.read_csv(f); d['date']=pd.to_datetime(d['date']); return d.drop_duplicates('date').set_index('date')['close'].astype(float)
    return None
raw={s:load(s) for s in U}; raw={s:v for s,v in raw.items() if v is not None}
px=pd.DataFrame(raw).sort_index(); r=px.pct_change(fill_method=None)
breadth=(r>0).mean(axis=1); mom=px.pct_change(10)
sig=pd.DataFrame(index=px.index,columns=raw,dtype=float)
hi=breadth.shift(1)>=.60; lo=breadth.shift(1)<=.40; mid=~(hi|lo)
sig.loc[hi]=mom.loc[hi]; sig.loc[lo]=-mom.loc[lo]; sig.loc[mid]=.5*mom.loc[mid]; sig=sig.shift(1)
fwd=px.pct_change(fill_method=None).shift(-1)
ics=[]; rows=[]; dates=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(q): ics.append(q); rows.append(len(z)); dates.append(dt)
ics=np.array(ics); D=pd.Series(dates)
def rep(m):
 a=ics[m]; return len(a),round(float(a.mean()),6),round(float(a.mean()/(a.std(ddof=1)+1e-12)),6),round(float((a>0).mean()),4)
print('assets',len(raw),'dates',len(ics),'rows',sum(rows),'avg_names',round(np.mean(rows),2),'coverage',round(sig.notna().sum().sum()/sig.size,4),'range',D.min(),D.max())
for n,m in [('full',np.ones(len(ics),bool)),('2020-22',D.dt.year<=2022),('2023-25',D.dt.year.between(2023,2025)),('2026',D.dt.year==2026),('2027',D.dt.year==2027),('last180',D>=D.max()-pd.Timedelta(days=180)),('last90',D>=D.max()-pd.Timedelta(days=90))]: print(n,rep(np.asarray(m)))
