import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-09-22')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); m=r.mean(axis=1)
# residual 3-day move versus equal-weight cross-asset market, all inputs lagged one day
res=(r.sub(m,axis=0)).rolling(3,min_periods=3).sum().shift(1); sig=-res
fwd=px.shift(-1)/px-1
def calc(S,F):
 a=[]; n=[]
 for d in S.index:
  g=pd.DataFrame({'s':S.loc[d],'f':F.loc[d]}).dropna()
  if len(g)>=8 and g.s.nunique()>1:a.append(spearmanr(g.s,g.f).statistic);n.append(len(g))
 a=np.array(a); return {'dates':len(a),'avg_n':round(np.mean(n),2) if n else None,'ic':round(np.mean(a),6) if len(a) else None,'icir':round(np.mean(a)/np.std(a,ddof=1),6) if len(a)>1 else None,'hit':round(np.mean(a>0),4) if len(a) else None}
print('end',px.index.max().date(),'dates',len(px),'coverage',round(sig.stack().notna().mean(),4),'overall',calc(sig,fwd)); z=sig.index.year
for k,c in [('2020-22',z<=2022),('2023-25',(z>=2023)&(z<=2025)),('2026',z==2026),('2027',z==2027),('last90',sig.index>=END-pd.Timedelta(days=90)),('last180',sig.index>=END-pd.Timedelta(days=180))]:print(k,calc(sig[c],fwd[c]))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20270923_residual_reversal_signal.csv',index=False)
