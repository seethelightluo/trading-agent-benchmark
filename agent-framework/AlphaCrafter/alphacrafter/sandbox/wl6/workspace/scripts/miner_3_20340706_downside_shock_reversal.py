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
ret5=px.pct_change(5); ret10=px.pct_change(10)
# Robust downside semideviation: RMS of negative daily returns, avoiding sparse rolling std NaNs.
down=np.sqrt((r.clip(upper=0)**2).rolling(30,min_periods=10).mean()).replace(0,np.nan)
shock=-ret10/(down*np.sqrt(10))
vol20=r.rolling(20,min_periods=10).std(); shock_gate=(1+0.45*(-ret5).clip(lower=0)/(vol20*np.sqrt(5))).clip(1,2.2)
factor=(shock*shock_gate).shift(1); factor.to_csv('scripts/miner_3_20340706_downside_shock_reversal_signal.csv',index_label='date')
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
fw=px.shift(-20)/px-1; rows=[]
for dt in factor.index:
 z=pd.concat([factor.loc[dt],fw.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1])))
q=pd.DataFrame(rows,columns=['date','ic']).set_index('date')
for a,b in [('2020','2024-12-31'),('2027','2029-12-31'),('2030','2032-12-31'),('2033','2034-12-31')]:
 x=q.loc[a:b].ic.dropna(); print('regime',a,b,'n',len(x),'IC',round(x.mean(),8) if len(x) else None)
print('turnover',round(factor.rank(pct=True).diff().abs().stack().mean(),6))
