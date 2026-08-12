import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,2600)
 if x is None or len(x)<100:x=get_index_daily_data(s,2600)
 if x is not None:D[s]=x.assign(date=pd.to_datetime(x.date)).set_index('date').sort_index()['close']
C=pd.DataFrame(D); r=C.pct_change();
try:
 m=pd.read_csv('../persistent/index_data/DXY.csv'); m['date']=pd.to_datetime(m['date']); m=m.set_index('date')['close'].reindex(C.index).ffill().pct_change()
except: m=pd.Series(0,index=C.index)
# residual medium trend: 15d return minus rolling beta to DXY * DXY return, lag one day
beta=r.rolling(60,min_periods=30).cov(m).div(m.rolling(60,min_periods=30).var(),axis=0)
f=(r.rolling(15).sum()-beta.mul(m.rolling(15).sum(),axis=0)).shift(1)
fwd=C.pct_change().shift(-1); ic=[]; ns=[]
for d in f.index:
 z=pd.concat([f.loc[d],fwd.loc[d]],axis=1).dropna()
 if len(z)>=8:ic.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
ic=pd.Series(ic).dropna(); print('candidate=dxy_residual_15d_trend','dates',len(ic),'avgN',np.mean(ns),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',np.mean(ic>0),'coverage',np.mean(ns)/len(U)); print('recent',ic.tail(504).mean(),ic.tail(504).mean()/ic.tail(504).std(ddof=1))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20290628_dxy_residual_15d_signal.csv',index=False)
