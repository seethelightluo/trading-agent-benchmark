import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-09-22'); P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); neg=r.where(r<0,0.0); dn=neg.rolling(20,min_periods=20).std(); tot=r.rolling(20,min_periods=20).std(); sig=-(dn/(tot+1e-12)).shift(1); fwd=px.shift(-1)/px-1

def calc(S,F,mask=None):
 a=[];n=[]
 for i,d in enumerate(S.index):
  if mask is not None and not bool(mask[i]): continue
  g=pd.DataFrame({'s':S.loc[d],'f':F.loc[d]}).dropna()
  if len(g)>=8 and g.s.nunique()>1:
   z=spearmanr(g.s,g.f).statistic
   if np.isfinite(z):a.append(z);n.append(len(g))
 a=np.array(a);return len(a),round(np.mean(n),2),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6),round((a>0).mean(),4)
print('end',px.index.max().date(),'overall',calc(sig,fwd)); y=sig.index.year
for q,c in [('2020-22',(y>=2020)&(y<=2022)),('2023-25',(y>=2023)&(y<=2025)),('2026',y==2026),('2027',y==2027),('last180',sig.index>=END-pd.Timedelta(days=180))]:print(q,calc(sig,fwd,c))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20270923_downside_asymmetry_signal.csv',index=False)
