import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
UNIV=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
allx={}
for s in UNIV:
    try: d=get_stock_daily_data(s,days=4000)
    except Exception: d=None
    if d is not None and len(d): allx[s]=d[['date','close']].drop_duplicates('date').set_index('date').close.rename(s)
for s in ['DXY','VIX']:
    try: d=get_index_daily_data(s,days=4000)
    except Exception: d=None
    if d is not None and len(d): allx[s]=d[['date','close']].drop_duplicates('date').set_index('date').close.rename(s)
px=pd.concat(allx,axis=1).sort_index().loc['2020-01-01':]
px=px.replace([np.inf,-np.inf],np.nan)
r=np.log(px).diff(); out=[]
for t in px.index:
    hist=r.loc[:t].tail(61)
    if len(hist)<55 or not {'DXY','VIX'}.issubset(hist.columns): continue
    vals={}
    for s in UNIV:
        if s not in r: continue
        h=hist[[s,'DXY','VIX']].replace([np.inf,-np.inf],np.nan).dropna()
        if len(h)<45: continue
        X=np.column_stack([np.ones(len(h)),h[['DXY','VIX']].values]); y=h[s].values
        b=np.linalg.lstsq(X,y,rcond=None)[0]
        rr20=r.loc[:t,s].tail(20).sum()-b[1]*r.loc[:t,'DXY'].tail(20).sum()-b[2]*r.loc[:t,'VIX'].tail(20).sum()
        vol=h[s].tail(40).std()
        if np.isfinite(rr20) and np.isfinite(vol) and vol>1e-8: vals[s]=-rr20/(vol*np.sqrt(20))
    if len(vals)>=8: out.append((t,vals))
sig=pd.DataFrame([{'date':t,'symbol':s,'signal':x} for t,v in out for s,x in v.items()])
fwd=np.log(px).shift(-1)-np.log(px); recs=[]
for t,g in sig.groupby('date'):
    y=fwd.loc[t].reindex(g.symbol).dropna(); z=g.set_index('symbol').signal.reindex(y.index).dropna(); y=y.reindex(z.index)
    if len(z)>=8 and y.notna().all(): recs.append((t,z.rank().corr(y.rank()),len(z)))
ic=pd.Series({t:x for t,x,n in recs}).sort_index()
print('endpoint',px.index.max().date(),'dates',len(ic),'avgN',round(np.mean([n for t,x,n in recs]),2),'coverage',round(len(sig)/(len(px)*len(UNIV)),4))
for label,q in [('full',ic),('2028+',ic[ic.index>='2028-01-01']),('2029+',ic[ic.index>='2029-01-01']),('2030',ic[ic.index>='2030-01-01'])]:
 print(label,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/(q.std(ddof=1)+1e-12),6),'hit',round(np.mean(q>0),4))
path='scripts/miner_2_20300711_macro_residual_reversal_signal.csv';sig.to_csv(path,index=False);print('artifact',path)
