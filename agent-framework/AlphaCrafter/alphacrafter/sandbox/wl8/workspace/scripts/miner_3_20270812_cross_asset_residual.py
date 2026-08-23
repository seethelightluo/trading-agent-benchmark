import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-08-11'); P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); x=x[x.date<=END].sort_values('date').set_index('date'); P[s]=x.close
px=pd.DataFrame(P); ret=px.pct_change(); r3=px.pct_change(3); meanr=r3.mean(axis=1); vol=ret.rolling(20,min_periods=15).std().shift(1)
sig=-(r3.sub(meanr,axis=0).shift(1)/vol); f1=px.shift(-1)/px-1; f5=px.shift(-5)/px-1
def calc(S,F):
 out=[];ns=[]
 for d in S.index:
  g=pd.DataFrame({'s':S.loc[d],'f':F.loc[d]}).dropna()
  if len(g)>=8 and g.s.nunique()>1 and g.f.nunique()>1:out.append(spearmanr(g.s,g.f).statistic);ns.append(len(g))
 v=np.array(out);return len(v),round(np.mean(ns),2),round(v.mean(),6),round(v.mean()/v.std(ddof=1),6),round(np.mean(v>0),4)
print('overall1',calc(sig,f1),'coverage',round(sig.stack().notna().mean(),4));print('overall5',calc(sig,f5))
y=sig.index.year
for q,c in [('2020-22',y<=2022),('2023-25',(y>=2023)&(y<=2025)),('2026',y==2026),('2027',y==2027),('last90',sig.index>=END-pd.Timedelta(days=135))]:print(q,calc(sig[c],f1))
sig.stack().rename('sig').reset_index().rename(columns={'level_1':'symbol'}).to_csv('scripts/miner_3_20270812_cross_asset_residual_signal.csv',index=False)
