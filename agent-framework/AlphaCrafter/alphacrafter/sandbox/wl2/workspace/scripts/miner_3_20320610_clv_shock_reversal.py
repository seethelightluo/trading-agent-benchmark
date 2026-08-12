import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    try:d=get_index_daily_data(s,days=5000)
    except:d=None
    if d is None or len(d)<100:
        try:d=get_stock_daily_data(s,days=5000)
        except:d=None
    if d is None:return None
    d=d.copy();d.date=pd.to_datetime(d.date);return d.sort_values('date').drop_duplicates('date').set_index('date')
D={s:fetch(s) for s in U};D={s:x for s,x in D.items() if x is not None}
C=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index(); R=C.pct_change()
# Close-location reversal: persistent closes near daily lows after a negative 5d move
H=pd.DataFrame({s:d.high.astype(float) for s,d in D.items()}).reindex(C.index)
L=pd.DataFrame({s:d.low.astype(float) for s,d in D.items()}).reindex(C.index)
clv=((C-L)/(H-L).replace(0,np.nan)-.5)
shock=C.pct_change(5)/(R.rolling(20).std()*np.sqrt(5)).replace(0,np.nan)
# low close is positive reversal signal; emphasize only materially negative shock
f=(-clv.rolling(3).mean()*(-shock).clip(lower=0)).replace([np.inf,-np.inf],np.nan)
# smooth cross-sectional standardization is not needed for ranks
f=f.sub(f.mean(axis=1),axis=0)
rows=[]
for d in f.index:
 q=pd.concat([f.loc[d],R.shift(-1).loc[d]],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:rows.append((d,q.iloc[:,0].rank().corr(q.iloc[:,1].rank()),len(q)))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('assets',len(D),'price_dates',len(C),'IC_dates',len(o),'avg_n',round(o.n.mean(),3),'coverage',round(o.n.mean()/len(U),4))
print('IC %.6f ICIR %.6f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(),(o.ic>0).mean()))
for a,b in [('2020','2022'),('2023','2025'),('2026','2029'),('2030','2032')]:
 q=o.loc[a:b].ic;print(a+'-'+b,'dates',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(),(q>0).mean()))
for h in [1,3,5,10]:
 rr=C.pct_change(h).shift(-h);v=[]
 for d in f.index:
  q=pd.concat([f.loc[d],rr.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:v.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 print('decay',h,'IC %.6f n %d'%(np.nanmean(v),len(v)))
print('recent120','IC %.6f ICIR %.6f n %d'%(o.tail(120).ic.mean(),o.tail(120).ic.mean()/o.tail(120).ic.std(),len(o.tail(120))))
f.to_csv('scripts/miner_3_20320610_clv_shock_reversal_signal.csv',index_label='date')
