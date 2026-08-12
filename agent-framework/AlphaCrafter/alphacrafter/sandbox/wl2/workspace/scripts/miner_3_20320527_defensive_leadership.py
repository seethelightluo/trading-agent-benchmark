import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def fetch(s):
    
    try: d=get_index_daily_data(s, days=5000)
    except Exception: d=None
    if d is None or len(d)<80:
        try: d=get_stock_daily_data(s, days=5000)
        except Exception: d=None
    if d is None: return None
    d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.sort_values('date').drop_duplicates('date')
    return d.set_index('date')['close'].astype(float)
P={s:fetch(s) for s in U}; P={s:x for s,x in P.items() if x is not None}
px=pd.DataFrame(P).sort_index(); r=px.pct_change()
# Defensive-leadership factor: 10d risk-adjusted relative strength, activated smoothly
# when the defensive basket (gold and two yield series) has positive 5d trend.
vol=r.rolling(20).std().replace(0,np.nan)
rs=px.pct_change(10).sub(px.pct_change(10).mean(axis=1),axis=0)
z=rs/((vol*np.sqrt(10)).replace(0,np.nan))
defs=r[['XAU','US10Y','CN10Y']].rolling(5).sum().mean(axis=1)
# continuous regime multiplier 0.35..1.65, no future information
mult=(0.35+1.30/(1+np.exp(-defs*35))).clip(.35,1.65)
f=z.mul(mult,axis=0)
# forward 1d returns and daily cross-sectional Spearman IC
out=[]
for d in f.index:
    a=f.loc[d]; b=r.shift(-1).loc[d]
    q=pd.concat([a,b],axis=1).dropna()
    if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
        out.append((d,q.iloc[:,0].rank().corr(q.iloc[:,1].rank()),len(q)))
o=pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
print('assets',len(P),'dates',len(px),'IC_dates',len(o),'avg_n',round(o.n.mean(),2),'coverage',round(o.n.mean()/len(U),4))
print('IC %.6f ICIR %.6f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(),(o.ic>0).mean()))
for a,b in [('2020','2022'),('2023','2025'),('2026','2029'),('2030','2032')]:
 q=o.loc[a:b].ic
 print(a+'-'+b,'dates',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std() if len(q)>1 else np.nan,(q>0).mean()))
for h in [1,3,5,10]:
 rr=px.pct_change(h).shift(-h)/1 # forward h return, same factor date
 vals=[]
 for d in f.index:
  q=pd.concat([f.loc[d],rr.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: vals.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 print('decay',h,round(float(np.nanmean(vals)),6),len(vals))
print('recent',o.tail(120).ic.mean(),o.tail(120).ic.mean()/o.tail(120).ic.std(),len(o.tail(120)))
# artifact for deterministic audit
f.to_csv('scripts/miner_3_20320527_defensive_leadership_signal.csv',index_label='date')
