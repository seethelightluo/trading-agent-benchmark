import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
    try: d=get_stock_daily_data(s, days=4000)
    except Exception as e: print('skip',s,e); continue
    if d is not None and len(d):
        x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date); x=x.dropna().drop_duplicates('date').set_index('date').sort_index(); px[s]=x.close
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); f=-r.rolling(20,min_periods=15).std()
def calc(h):
    vals=[]
    for i in range(len(p)-h-1):
        nxt=p.iloc[i+h]/p.iloc[i]-1; q=pd.concat([f.iloc[i],nxt],axis=1,keys=['f','y']).dropna()
        if len(q)>=8: vals.append((p.index[i],len(q),q.f.corr(q.y)))
    a=pd.DataFrame(vals,columns=['date','n','ic']).set_index('date'); m=a.ic.mean(); ir=m/a.ic.std(ddof=1)
    return a,m,ir
for h in [1,5,10,20]:
    a,m,ir=calc(h); print(f'h{h}: dates={len(a)} avgN={a.n.mean():.2f} IC={m:.6f} ICIR={ir:.6f} hit={(a.ic>0).mean():.3f}')
    for lo,hi in [('2020','2022'),('2023','2025'),('2026','2028')]:
        z=a.loc[lo:hi]
        if len(z): print(' ',lo+'-'+hi,len(z),f'IC={z.ic.mean():.6f}',f'ICIR={z.ic.mean()/z.ic.std(ddof=1):.6f}')
f2=f.rank(axis=1,pct=True); samp=f2.dropna(how='all').iloc[::10]; turn=[]
for j in range(1,len(samp)):
 aa=samp.iloc[j-1].dropna(); bb=samp.iloc[j].dropna(); ix=aa.index.intersection(bb.index); turn.append(np.mean((aa[ix]-bb[ix]).abs()))
print('coverage',f.notna().sum(axis=1).mean()/len(U),'turnover_rank_distance',np.mean(turn),'n_dates',len(p),'n_assets',len(px),'period',p.index.min(),p.index.max())
