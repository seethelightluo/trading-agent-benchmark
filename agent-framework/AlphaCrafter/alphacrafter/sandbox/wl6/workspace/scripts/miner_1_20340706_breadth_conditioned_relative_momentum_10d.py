import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    for f in (get_index_daily_data,get_stock_daily_data):
        try:
            d=f(s,days=4000)
            if d is not None and len(d): return d
        except Exception: pass
S={}
for s in U:
    d=fetch(s)
    if d is not None:
        d=d.copy(); d.date=pd.to_datetime(d.date); S[s]=d.set_index('date')
px=pd.DataFrame({s:x.close.astype(float) for s,x in S.items()}).sort_index(); r=px.pct_change()
# Relative 10-day momentum, conditioned on market breadth: damp continuation
# when fewer assets participate, and emphasize when broad participation confirms trend.
r10=px.pct_change(10); rel=r10.sub(r10.median(axis=1),axis=0)
breadth=(r.rolling(20).sum()>0).mean(axis=1)
# centered, bounded breadth confirmation; all inputs lagged one completed day
breadth_gate=(0.55+0.90*breadth.clip(.1,.9)).shift(1)
factor=rel.mul(breadth_gate,axis=0).shift(1)
factor.to_csv('scripts/miner_1_20340706_breadth_conditioned_relative_momentum_10d_signal.csv',index_label='date')
print('assets_loaded',px.shape[1],'dates',len(px),'cutoff',px.index.max().date())
for h in [5,10,20,40]:
    fw=px.shift(-h)/px-1; rows=[]
    for dt in factor.index:
        z=pd.concat([factor.loc[dt],fw.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
        if len(z)>=8:
            c=z.iloc[:,0].corr(z.iloc[:,1])
            if np.isfinite(c): rows.append((dt,c,len(z)))
    q=pd.DataFrame(rows,columns=['date','ic','n']); a=q.ic.to_numpy(); m=a.mean(); ir=m/a.std(ddof=1)*np.sqrt(len(a))
    print('h',h,'dates',len(a),'avg_names',round(q.n.mean(),3),'coverage',round(q.n.mean()/15,4),'IC',round(m,8),'ICIR',round(ir,6),'hit',round((a>0).mean(),4))
print('turnover',round(factor.rank(pct=True).diff().abs().stack().mean(),6))
for lo,hi in [('2020','2024-12-31'),('2027','2029-12-31'),('2030','2032-12-31'),('2033','2034-07-05')]:
    q=[]; fw=px.shift(-10)/px-1
    for dt in factor.loc[lo:hi].index:
        z=pd.concat([factor.loc[dt],fw.loc[dt]],axis=1).dropna()
        if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1]))
    print('regime',lo,hi,'n',len(q),'IC',round(float(np.nanmean(q)),8) if q else None)
