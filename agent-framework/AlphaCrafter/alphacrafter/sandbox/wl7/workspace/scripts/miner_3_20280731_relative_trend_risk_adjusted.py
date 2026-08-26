import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2028-07-30')
def load(s):
    for fn in (get_index_daily_data,get_stock_daily_data):
        try:
            x=fn(s,3000)
            if x is not None and len(x):
                x=x.copy(); x.date=pd.to_datetime(x.date).dt.normalize()
                return x.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
        except Exception: pass
D={s:load(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
# Relative trend: asset's 20d return in excess of contemporaneous cross-asset median,
# normalized by its 20d volatility and lagged one session.
rets=[]
for s,d in D.items():
 c=pd.to_numeric(d.close,errors='coerce'); r20=c.pct_change(20); vol=c.pct_change().rolling(20,min_periods=15).std()*np.sqrt(252)
 rets.append(pd.DataFrame({'date':c.index,'asset':s,'r20':r20,'vol':vol,'fr1':c.shift(-1)/c-1,'fr5':c.shift(-5)/c-1,'fr10':c.shift(-10)/c-1,'fr20':c.shift(-20)/c-1}))
x=pd.concat(rets,ignore_index=True)
x['med']=x.groupby('date')['r20'].transform('median')
x['f']=((x.r20-x.med)/(x.vol+0.02)).shift(1)
q=x.replace([np.inf,-np.inf],np.nan).dropna(subset=['f','fr1'])
def stat(x,col):
 z=[]; ns=[]
 for _,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g[col].nunique()>1:
   z.append(g.f.corr(g[col],method='spearman')); ns.append(len(g))
 z=pd.Series(z).dropna()
 return len(z),float(np.mean(ns)),float(z.mean()),float(z.mean()/z.std(ddof=1)*np.sqrt(252)),float((z>0).mean())
print('assets',len(D),'dates',q.date.nunique(),'avg_n',q.groupby('date').size().mean(),'coverage',len(q)/(q.date.nunique()*len(D)))
for col in ['fr1','fr5','fr10','fr20']: print(col,stat(q,col))
for a,b in [(2020,2022),(2023,2024),(2025,2026),(2027,2028)]: print('regime',a,b,stat(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)],'fr10'))
p=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',float(p.diff().abs().mean().mean()))
q.to_csv('scripts/miner_3_20280731_relative_trend_risk_adjusted_signal.csv',index=False)
