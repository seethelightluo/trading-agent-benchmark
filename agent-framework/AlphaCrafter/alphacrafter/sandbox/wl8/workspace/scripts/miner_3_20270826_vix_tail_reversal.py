import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-08-25'); P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P); r=px.pct_change(); r3=px.pct_change(3)
v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); v=v[v.date<=END].set_index('date').close.sort_index()
vr=v.pct_change(5); z=((vr-vr.rolling(60,min_periods=30).mean())/vr.rolling(60,min_periods=30).std()).shift(1).clip(-2,2)
base=-(r3-r3.mean(axis=1).values[:,None]); sig=base*(1+0.5*z.reindex(px.index).ffill()).values[:,None]
f=px.shift(-1)/px-1
def calc(S,F):
 a=[];n=[]
 for d in S.index:
  g=pd.DataFrame({'s':S.loc[d],'f':F.loc[d]}).dropna()
  if len(g)>=8 and g.s.nunique()>1:a.append(spearmanr(g.s,g.f).statistic);n.append(len(g))
 a=np.array(a);return len(a),round(np.mean(n),2),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6),round((a>0).mean(),4)
print('overall',calc(sig,f),'coverage',round(sig.stack().notna().mean(),4)); y=sig.index.year
for q,c in [('2020-22',y<=2022),('2023-25',(y>=2023)&(y<=2025)),('2026',y==2026),('2027',y==2027),('last180',sig.index>=END-pd.Timedelta(days=180))]:print(q,calc(sig[c],f[c]))
