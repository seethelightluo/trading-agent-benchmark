import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=None
    for fn in (get_index_daily_data,get_stock_daily_data):
        try: x=fn(s,days=5000)
        except Exception: pass
        if x is not None and len(x): break
    if x is not None: D[s]=x.set_index('date')['close'].astype(float)
P=pd.DataFrame(D).sort_index().ffill()
# delayed reversal: use 3-day return ending two sessions before signal date; cross-sectional demean and invert
rows=[]
for i in range(25,len(P)-10):
    d=P.index[i]
    # lagged: signal uses prices through i-2, forward starts i+1
    r3=P.iloc[i-2]/P.iloc[i-5]-1
    sig=-(r3-r3.median())
    for h in (3,5,10):
        f=P.iloc[i+h]/P.iloc[i]-1
        z=pd.concat([sig,f],axis=1).dropna()
        if len(z)>=8: rows.append((d,h,len(z),z.iloc[:,0].corr(z.iloc[:,1]),sig))
R=pd.DataFrame([(a,b,c,d) for a,b,c,d,e in rows],columns=['date','h','n','ic']).set_index('date')
for h in (3,5,10):
 q=R[R.h==h].ic.dropna(); print('H',h,'dates',len(q),'avgN',round(R[R.h==h].n.mean(),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 for label,mask in [('early',q.index<'2027-01-01'),('mid',(q.index>='2027-01-01')&(q.index<'2031-01-01')),('recent',q.index>='2031-01-01')]:
  z=q[mask]; print(label,len(z),round(z.mean(),6) if len(z) else None,round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
# turnover and coverage at 5d signal dates
S=[]; cov=[]
for i in range(25,len(P)-10):
 r=P.iloc[i-2]/P.iloc[i-5]-1; s=-(r-r.median()); S.append(s.rank(pct=True)); cov.append(s.notna().mean())
A=pd.DataFrame(S,index=P.index[25:len(P)-10]); print('coverage',np.mean(cov),'rank_turnover',A.diff().abs().mean().mean())
# save reproducible signal artifact at all dates/horizons
out=[]
for i in range(25,len(P)-10):
 d=P.index[i]; r=P.iloc[i-2]/P.iloc[i-5]-1; s=-(r-r.median())
 for a in U:
  if a in s.index and pd.notna(s[a]): out.append({'date':d,'asset':a,'signal':float(s[a])})
pd.DataFrame(out).to_csv('scripts/miner_3_20340123_delayed_reversal_signal.csv',index=False)
