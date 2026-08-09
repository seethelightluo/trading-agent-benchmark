import pandas as pd, numpy as np, json, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 px[s]=d
macro=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
# exact date aligned returns
rets=pd.DataFrame({s:x.pct_change() for s,x in px.items()}).join(macro.pct_change().rename('DXY'),how='inner')
# only common historical through research cutoff
rets=rets.loc[:'2026-07-15']
# rolling covariance / variance, trailing completed session at date t
cov=rets[U].rolling(60,min_periods=45).cov(rets['DXY']).unstack()[ 'DXY'] if False else None
# explicit safe calculation
v=rets['DXY'].rolling(60,min_periods=45).var()
f=pd.DataFrame(index=rets.index)
for s in U:
 f[s]=-rets[s].rolling(60,min_periods=45).cov(rets['DXY'])/v
# factor at t and forward return t+1
f=f.replace([np.inf,-np.inf],np.nan)
ics=[]; dates=[]; nms=[]
for dt in f.index:
 y=rets[U].shift(-1).loc[dt]
 z=pd.concat([f.loc[dt],y],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); nms.append(len(z))
ic=np.array(ics); print('dates',len(ic),'avg_names',np.mean(nms),'coverage',np.mean(nms)/15,'IC',np.nanmean(ic),'ICIR',np.nanmean(ic)/np.nanstd(ic,ddof=1),'hit',np.mean(ic>0),'turnover',np.nan)
# horizon 5/10 forward compounded, factor date
for h in [5,10]:
 ys=rets[U].rolling(h).sum().shift(-h+1) # t through t+h-1 includes current; adjust use t+1 onward
 ys=rets[U].shift(-1).rolling(h).sum()
 a=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],ys.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(a); print(h,'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'N',len(a))
for yr in range(2020,2027):
 a=ic[[d.year==yr for d in dates]]
 if len(a):print('regime',yr,len(a),np.mean(a),np.mean(a)/np.std(a,ddof=1))
# rank turnover
r=f.rank(axis=1,pct=True); print('rank_turn',r.diff().abs().mean().mean())
# correlation with effective factor signal artifacts pooled
for fn in os.listdir('factors'):
 if fn.endswith('.json') and not fn.endswith('.bak'):
  try:
   q=json.load(open('factors/'+fn)); print('LIB',fn,q.get('factor_id'))
  except:pass
