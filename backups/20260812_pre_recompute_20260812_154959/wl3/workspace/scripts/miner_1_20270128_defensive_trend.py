import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2027-01-27'
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().loc[:cut]
r=P.pct_change(); vol=r.rolling(40,min_periods=20).std()*np.sqrt(20)
# Defensive trend confirmation: medium/long risk-adjusted trend, rewarding persistent trends and penalizing disagreement.
m20=P.pct_change(20)/(vol+1e-8); m60=P.pct_change(60)/(vol+1e-8); m120=P.pct_change(120)/(vol+1e-8)
f=(0.25*m20+0.45*m60+0.30*m120)
# require agreement of medium and long trend, otherwise damp signal
agree=((P.pct_change(20)*P.pct_change(60)>0)&(P.pct_change(60)*P.pct_change(120)>0)).astype(float)
f=f*(0.5+0.5*agree)
f=f.replace([np.inf,-np.inf],np.nan)
def calc(y, dates):
 q=[]; ns=[]
 for d in dates:
  z=pd.DataFrame({'f':f.loc[d],'y':y.loc[d]}).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
 q=np.asarray(q); return q,ns
for lab,y in [('1d',P.pct_change().shift(-1)),('3d',P.pct_change(3).shift(-3)),('5d',P.pct_change(5).shift(-5)),('10d',P.pct_change(10).shift(-10))]:
 q,n=calc(y,f.index);print(lab,'dates',len(q),'avgN',round(np.mean(n),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
for a,b in [('2020-01','2022-12'),('2023-01','2024-12'),('2025-01','2027-01')]:
 q,n=calc(P.pct_change().shift(-1),f.loc[a:b].index);print('regime',a,b,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),4),'period',P.index.min().date(),P.index.max().date())
f.to_csv('scripts/miner_1_20270128_defensive_trend_signal.csv')
