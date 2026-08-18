import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2028-09-12'); base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv')); d['date']=pd.to_datetime(d.date); px[s]=d.sort_values('date').set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().loc[:cutoff]; r=P.pct_change(); v=r.rolling(20,min_periods=15).std().shift(1)
# Causal adaptive shock reversal: stronger reversal when prior cross-sectional dispersion is elevated,
# blended with slow trend recovery when dispersion is quiet. All components lagged.
shock=(-r.pct_change(3)/(v+1e-12)).shift(1)
trend=P.pct_change(20).shift(1)/(v+1e-12)
disp=r.rolling(20,min_periods=15).std().mean(axis=1).shift(1)
med=disp.rolling(60,min_periods=30).median().shift(1)
# bounded regime weight, no future data
w=(disp/(med+1e-12)).clip(0.5,2.0).sub(0.5).div(1.5).clip(0,1)
f=(-(w.values[:,None])*0) if False else (w.to_numpy()[:,None]*shock.to_numpy()+(1-w.to_numpy()[:,None])*trend.to_numpy())
f=pd.DataFrame(f,index=P.index,columns=P.columns).replace([np.inf,-np.inf],np.nan)
fr=P.shift(-10)/P-1
rows=[]
for dt in P.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
o=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); q=o.ic
def stats(x): return len(x),x.mean(),x.mean()/x.std(ddof=1),float((x>0).mean())
print('candidate adaptive_dispersion_shock_trend universe',len(U),'dates',len(o),'meanN',o.n.mean(),'coverage',o.n.mean()/len(U))
print('overall n IC ICIR hit',stats(q))
for name,x in [('2020-22',q.loc[:'2022-12-31']),('2023-24',q.loc['2023-01-01':'2024-12-31']),('2025-26',q.loc['2025-01-01':'2026-12-31']),('2027-28',q.loc['2027-01-01':]),('recent60',q.tail(60)),('recent120',q.tail(120)),('recent252',q.tail(252))]:
 print(name,stats(x))
ranks=f.rank(axis=1,pct=True); turn=(ranks-ranks.shift(1)).abs().mean(axis=1); print('turnover_proxy',turn.loc[o.index].mean())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20280912_adaptive_dispersion_signal.csv',index=False)
o.to_csv('scripts/miner_2_20280912_adaptive_dispersion_ic.csv')
print('artifact scripts/miner_2_20280912_adaptive_dispersion_signal.csv')
