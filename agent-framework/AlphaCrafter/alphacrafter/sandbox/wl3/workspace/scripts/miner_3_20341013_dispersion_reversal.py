import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index(); P=P.loc[P.index<=pd.Timestamp('2034-10-13')]; R=P.pct_change(); vol=R.rolling(20).std(); disp=R.median(axis=1).rolling(20).std()
# Cross-sectional 5d reversal, scaled by own realized vol and activated when market dispersion is elevated.
raw=-P.pct_change(5).div(vol); state=(disp>disp.rolling(120,min_periods=60).median()).astype(float); F=raw.mul(state,axis=0).shift(1)
print('assets',len(px),'dates',len(P),'avg_valid',round(F.notna().sum(axis=1).mean(),3))
for h in [1,3,5,10,20]:
 a=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(a); print('h',h,'n',len(a),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/np.nanstd(a,ddof=1)*np.sqrt(len(a)),4),'hit',round(np.mean(a>0),4))
# recent required horizon
for n in [120,252,756]:
 q=a[-n:]; print('recent20',n,'IC',round(np.nanmean(q),6),'ICIR',round(np.nanmean(q)/np.nanstd(q,ddof=1)*np.sqrt(len(q)),4))
print('turnover',round(np.nanmean(np.abs(F.rank(axis=1,pct=True)-F.rank(axis=1,pct=True).shift()).mean(axis=1)),4))
