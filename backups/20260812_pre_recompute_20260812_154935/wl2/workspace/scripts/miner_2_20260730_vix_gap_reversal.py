import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15']
# Macro-conditioned overnight gap reversal: fade the completed-session gap more aggressively
# when VIX is elevated relative to its trailing 60-session median; all inputs are lagged by construction.
gap=pd.DataFrame({s:-(D[s].open/D[s].close.shift(1)-1) for s in U})
vix=v.close.reindex(gap.index).ffill()
reg=(vix/vix.rolling(60,min_periods=30).median()).clip(0.5,2.0)
F=gap.mul(reg,axis=0)
Y=pd.DataFrame({s:D[s].close.shift(-1)/D[s].close-1 for s in U})
q=[]; ns=[]
for dt in F.index:
 z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
 if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
q=np.asarray(q)
print('horizon 1 dates',len(q),'meanN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
for h in [5,10]:
 yh=pd.DataFrame({s:D[s].close.shift(-h)/D[s].close-1 for s in U});qq=[];nn=[]
 for dt in F.index:
  z=pd.DataFrame({'f':F.loc[dt],'y':yh.loc[dt]}).dropna()
  if len(z)>=8:qq.append(spearmanr(z.f,z.y).statistic);nn.append(len(z))
 qq=np.asarray(qq);print('horizon',h,'dates',len(qq),'meanN',round(np.mean(nn),2),'IC',round(qq.mean(),6),'ICIR',round(qq.mean()/qq.std(ddof=1),6))
for yr in range(2020,2027):
 x=[]
 for dt in F.loc[str(yr)].index:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
  if len(z)>=8:x.append(spearmanr(z.f,z.y).statistic)
 print('regime',yr,'dates',len(x),'IC',round(np.mean(x),6) if x else None)
print('coverage',round(F.notna().sum().sum()/F.size,4),'rank_turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for k in [252,504,756]:
 x=q[-k:];print('recent',k,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'dates',len(x))
