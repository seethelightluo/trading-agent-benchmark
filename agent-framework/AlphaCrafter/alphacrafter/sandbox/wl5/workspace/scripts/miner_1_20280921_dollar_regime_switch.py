import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close']
dates=sorted(set.intersection(*[set(x.index) for x in px.values()]) & set(dxy.index))
P=pd.DataFrame({a:px[a].reindex(dates) for a in assets}); D=dxy.reindex(dates).ffill()
ret=P.pct_change(); mom=P.pct_change(20); vol=ret.rolling(60).std()*np.sqrt(20)
dollar=D/D.rolling(60).mean()-1
base=mom/vol
F=base.mul(np.where(dollar.to_numpy()<=0,1.0,-1.0),axis=0)
rows=[]
for i in range(60,len(dates)-10):
 x=F.iloc[i]; y=P.iloc[i+10]/P.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((dates[i],spearmanr(x[ok],y[ok]).statistic,ok.sum()))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
def stats(z): return float(z.mean()),float(z.mean()/z.std(ddof=1)),float((z>0).mean()),len(z)
print('dates',len(r),'avg_n',r.n.mean(),'coverage',r.n.sum()/(len(r)*15)); print('all IC ICIR hit n',stats(r.ic))
for name,z in [('2020-24',r.loc['2020':'2024'].ic),('2025-26',r.loc['2025':'2026'].ic),('2027-28',r.loc['2027':'2028'].ic),('recent252',r.ic.tail(252))]: print(name,stats(z))
print('turnover',float(F.rank(axis=1,pct=True).diff().abs().mean().mean()))
for h in [5,10,20]:
 rr=[]
 for i in range(60,len(dates)-h):
  x=F.iloc[i]; y=P.iloc[i+h]/P.iloc[i]-1; ok=x.notna()&y.notna()
  if ok.sum()>=8: rr.append(spearmanr(x[ok],y[ok]).statistic)
 print('decay',h,stats(pd.Series(rr)))
out=F.copy(); out.insert(0,'date',dates); out.to_csv('scripts/miner_1_20280921_dollar_regime_switch_signal.csv',index=False)
