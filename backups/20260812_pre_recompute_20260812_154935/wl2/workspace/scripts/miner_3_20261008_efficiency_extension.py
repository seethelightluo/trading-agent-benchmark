import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
end=min(max(x.index.max() for x in D.values()),pd.Timestamp('2026-10-07')); dates=D['SPX'].index[(D['SPX'].index>='2020-04-01')&(D['SPX'].index<=end)]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); r=C.pct_change()
# Stable directional trend: 20d signed efficiency, blended with 5d return; lagged one day.
e20=(C/C.shift(20)-1)/(20*r.abs().rolling(20,min_periods=15).mean()+.01)
m5=C/C.shift(5)-1
F=(e20.rank(axis=1,pct=True)+.35*m5.rank(axis=1,pct=True)).rank(axis=1,pct=True).shift(1)
def met(x): return (len(x),float(x.mean()),float(x.mean()/x.std(ddof=1)),float((x>0).mean()))
for h in [1,3,5,10]:
 Y=C.shift(-h).div(C)-1;q=[];n=[]
 for dt in dates:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);n.append(len(z))
 q=np.array(q);print('horizon',h,'dates',len(q),'avgN',round(np.mean(n),2),'IC ICIR hit',*[round(v,6) for v in met(q)[1:]])
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'end',end.date())
for k in [63,126,252,504]:print('recent',k,met(q[-k:]))
print('artifact formula=rank(rank(e20)+0.35*rank(return5)); e20=return20/(20*mean(abs(daily_return),20)+0.01); lag=1')
