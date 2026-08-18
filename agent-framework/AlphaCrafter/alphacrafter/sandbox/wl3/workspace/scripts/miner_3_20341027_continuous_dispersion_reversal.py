import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index(); P=P.loc[P.index<=pd.Timestamp('2034-10-27')]; R=P.pct_change()
# Continuous dispersion-weighted short-term reversal: 5d reversal / 20d vol, weighted by
# clipped standardized 20d cross-asset dispersion, lagged one day (no sparse activation).
vol=R.rolling(20).std(); csdisp=R.std(axis=1).rolling(20).mean(); base=csdisp.rolling(120,min_periods=60)
z=(csdisp-base.mean())/base.std(); gate=z.clip(-1,2).fillna(0)
F=(-P.pct_change(5).div(vol)).mul(gate,axis=0).shift(1)
print('assets',len(U),'dates',len(P),'avg_valid',round(F.notna().sum(axis=1).mean(),3))
for h in [1,3,5,10,20]:
 a=[]
 for i in range(len(P)-h):
  q=pd.concat([F.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(q)>=8:a.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 a=np.array(a); print('h',h,'n',len(a),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/np.nanstd(a,ddof=1)*np.sqrt(len(a)),4),'hit',round(np.mean(a>0),4))
 if h==10:
  for n in [120,252,756]:
   q=a[-n:]; print('recent',n,'IC',round(np.nanmean(q),6),'ICIR',round(np.nanmean(q)/np.nanstd(q,ddof=1)*np.sqrt(len(q)),4))
# rank turnover among valid signals
rk=F.rank(axis=1,pct=True); print('turnover',round(np.nanmean(np.abs(rk-rk.shift()).mean(axis=1)),4))
