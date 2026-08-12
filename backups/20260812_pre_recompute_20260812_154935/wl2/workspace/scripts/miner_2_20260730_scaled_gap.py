import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
# Volatility-scaled overnight gap reversal. At t, use t-1 observed open/prior close gap,
# scaled by trailing 20d close-return volatility known at t-1; predict t+1 close return.
F={}; Y={}
for s,x in D.items():
 gap=(x.open/x.close.shift(1)-1).shift(1)
 vol=x.close.pct_change().rolling(20,min_periods=15).std().shift(1)
 F[s]=-gap/vol.replace(0,np.nan)
 Y[s]=x.close.shift(-1)/x.close-1
F=pd.DataFrame(F).sort_index(); Y=pd.DataFrame(Y).sort_index()
def calc(y):
 q=[];ns=[]
 for dt in F.index:
  z=pd.DataFrame({'f':F.loc[dt],'y':y.loc[dt]}).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
 a=np.asarray(q);return len(a),np.mean(ns),np.mean(a),np.mean(a)/np.std(a,ddof=1),(a>0).mean()
print('daily dates meanN IC ICIR hit',calc(Y))
for h in [5,10]:
 yh=pd.DataFrame({s:D[s].close.shift(-h)/D[s].close-1 for s in U}).sort_index();print('horizon',h,calc(yh))
for y in range(2020,2027):
 q=[]
 for dt in F.loc[str(y)].index:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic)
 a=np.asarray(q);print('regime',y,len(a),round(a.mean(),6) if len(a) else None,round(a.mean()/a.std(ddof=1),6) if len(a)>1 else None)
print('rank turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'cell coverage',F.notna().sum().sum()/F.size)
for name,old in [('reversal5',-pd.DataFrame({s:D[s].close.pct_change(5) for s in U})),('momentum20',pd.DataFrame({s:D[s].close.pct_change(20) for s in U}))]:
 z=pd.concat([F.stack().rename('new'),old.stack().rename('old')],axis=1).dropna();print('corr',name,z.corr().iloc[0,1])
