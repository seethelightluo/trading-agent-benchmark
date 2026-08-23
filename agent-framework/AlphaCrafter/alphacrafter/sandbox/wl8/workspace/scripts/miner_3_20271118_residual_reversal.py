import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-11-17'); P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date)
 P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change()
# Lagged idiosyncratic return: remove the prior-day cross-asset equal-weight move,
# then reverse the residual. All inputs are observable before the forecast day.
lag=r.shift(1); common=lag.mean(axis=1); residual=lag.sub(common,axis=0)
sig=-residual
fwd=px.shift(-1)/px-1

def calc(mask=None):
 a=[]; ns=[]
 for i,d in enumerate(sig.index):
  if mask is not None and not bool(mask[i]): continue
  g=pd.DataFrame({'s':sig.loc[d],'f':fwd.loc[d]}).dropna()
  if len(g)>=8 and g.s.nunique()>1:
   v=spearmanr(g.s,g.f).statistic
   if np.isfinite(v): a.append(v); ns.append(len(g))
 a=np.asarray(a)
 if len(a)<2:return len(a),None,None,None,None
 return len(a),round(float(np.mean(ns)),2),round(float(a.mean()),6),round(float(a.mean()/a.std(ddof=1)),6),round(float((a>0).mean()),4)
y=sig.index.year
print('end',px.index.max().date(),'overall',calc())
for q,m in [('2020-22',(y>=2020)&(y<=2022)),('2023-25',(y>=2023)&(y<=2025)),('2026',y==2026),('2027',y==2027),('last180',sig.index>=END-pd.Timedelta(days=180))]:print(q,calc(m))
print('coverage',int(sig.notna().sum().sum()),'of',sig.size,'dates',sig.index.size)
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20271118_residual_reversal_signal.csv',index=False)
