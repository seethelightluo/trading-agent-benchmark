import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-09-08'); P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); sig=-r.rolling(20,min_periods=20).std().shift(1); fwd=px.shift(-1)/px-1
def calc(S,F):
 a=[];n=[]
 for d in S.index:
  g=pd.DataFrame({'s':S.loc[d],'f':F.loc[d]}).dropna()
  if len(g)>=8 and g.s.nunique()>1:a.append(spearmanr(g.s,g.f).statistic);n.append(len(g))
 a=np.array(a)
 if len(a)<2:return len(a),None,None,None
 return len(a),round(np.mean(n),2),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6)
print('end',px.index.max().date(),'coverage',round(sig.stack().notna().mean(),4),'overall',calc(sig,fwd)); y=sig.index.year
for q,c in [('2026',y==2026),('2027',y==2027),('last180',sig.index>=END-pd.Timedelta(days=180))]:print(q,calc(sig[c],fwd[c]))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20270909_lowvolatility_signal.csv',index=False)
