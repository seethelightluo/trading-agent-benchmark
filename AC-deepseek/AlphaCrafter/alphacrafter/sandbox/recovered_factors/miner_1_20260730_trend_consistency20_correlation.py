"""Complete required library-correlation evidence for miner_1 trend-consistency candidate."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; p={};v={}
for a in A:
 d=get_stock_daily_data(a,5000).copy();d.date=pd.to_datetime(d.date);d=d.drop_duplicates('date').set_index('date').sort_index();p[a]=d.close.astype(float);v[a]=d.volume.astype(float)
r=pd.DataFrame({a:x.pct_change(fill_method=None) for a,x in p.items()})
new=(r>0).astype(float).rolling(20,min_periods=15).mean().where(r.notna().rolling(20,min_periods=15).sum()>=15)
old={
 'miner_3_risk_adjusted_trend_20d':pd.DataFrame({a:(p[a]/p[a].shift(20)-1)/r[a].rolling(20,min_periods=15).std() for a in A}),
 'miner_1_ravmom_20obs':pd.DataFrame({a:(p[a]/p[a].shift(20)-1)/r[a].rolling(20,min_periods=15).std() for a in A}),
 'miner_1_volnorm_reversal_5obs':pd.DataFrame({a:-(p[a]/p[a].shift(5)-1)/r[a].rolling(5,min_periods=4).std() for a in A}),
 'miner_2_realized_volatility_20obs':pd.DataFrame({a:r[a].rolling(20,min_periods=15).std() for a in A}),
 'miner_3_relative_volume_participation_20d':pd.DataFrame({a:np.log(v[a]/v[a].rolling(20,min_periods=15).mean()) for a in A})}
vals=[]
for name,x in old.items():
 z=pd.concat([new.stack().rename('new'),x.stack().rename('old')],axis=1).dropna();rho=spearmanr(z.new,z.old).statistic;vals.append(abs(rho));print(name,'pairs',len(z),'rho',rho)
print('MAX_ABS_LIBRARY_CORRELATION',max(vals))
