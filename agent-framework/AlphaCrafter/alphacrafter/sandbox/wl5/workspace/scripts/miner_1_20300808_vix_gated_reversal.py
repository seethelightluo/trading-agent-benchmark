import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data/'
px={s:pd.read_csv(base+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}
wide=pd.DataFrame(px).sort_index(); ret=wide.pct_change()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(wide.index).ffill()
# recent loser reversal, gated by volatility regime: emphasize reversal when VIX is rising
r5=wide.pct_change(5); vol20=ret.rolling(20).std(); vol60=ret.rolling(60).std()
basefac=(-r5.clip(upper=0))/vol20 * np.sqrt((vol60/vol20).clip(lower=.25,upper=4))
vixmom=vix.pct_change(10)
gate=(1+vixmom.clip(lower=-.5,upper=.5)).clip(lower=.5,upper=1.5)
fac=basefac.mul(gate,axis=0)
rows=[]
for i in range(len(wide)-10):
 d=wide.index[i]; f=fac.iloc[i]; fr=wide.iloc[i+10]/wide.iloc[i]-1
 z=pd.concat([f,fr],axis=1).dropna()
 if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(x),'mean_n',x.n.mean(),'coverage',x.n.mean()/15)
print('IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(),'hit',(x.ic>0).mean())
for a,b in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2029-12-31'),('2030','2030-08-07')]:
 y=x.loc[a:b].ic
 print(a,'n',len(y),'ic',y.mean(),'icir',y.mean()/y.std() if len(y)>1 else np.nan)
# rank turnover
r=fac.rank(axis=1,pct=True); print('turnover',r.diff().abs().mean(axis=1).mean())
# decay
for h in [5,10,20]:
 rr=[]
 for i in range(len(wide)-h):
  z=pd.concat([fac.iloc[i],(wide.iloc[i+h]/wide.iloc[i]-1)],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('horizon',h,'ic',np.nanmean(rr),'icir',np.nanmean(rr)/np.nanstd(rr))
# artifact current/all signals
out=fac.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_1_20300808_vix_gated_reversal_signal.csv',index=False)
