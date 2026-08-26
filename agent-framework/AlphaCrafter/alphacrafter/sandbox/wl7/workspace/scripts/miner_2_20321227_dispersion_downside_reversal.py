import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p); x['date']=pd.to_datetime(x.date); D[s]=x.set_index('date').close
px=pd.DataFrame(D).sort_index(); rets=px.pct_change()
# candidate: residual 5d reversal, normalized by downside vol, only when broad cross-asset dispersion is elevated
r5=px.pct_change(5); resid=r5.sub(r5.median(axis=1),axis=0)
down=rets.where(rets<0).rolling(30,min_periods=15).std()
base=-resid/(down+1e-8)
disp=rets.std(axis=1).rolling(60,min_periods=30).mean()
gate=(disp>disp.rolling(120,min_periods=60).median()).astype(float)
f=base.mul(gate,axis=0).replace([np.inf,-np.inf],np.nan)
# avoid all zeros on inactive dates by mark nan
f[gate==0]=np.nan
for h in [1,5,10,20]:
 fr=px.shift(-h)/px-1
 vals=[]; ns=[]; dates=[]
 for dt in f.index:
  a=f.loc[dt]; b=fr.loc[dt]
  z=pd.concat([a,b],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(dt)
 v=np.array(vals); print('H',h,'dates',len(v),'avgN',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(np.nanmean(v),np.nanmean(v)/(np.nanstd(v,ddof=1)+1e-12),np.mean(v>0)))
# turnover among active dates
rr=f.rank(axis=1,pct=True); active=rr.dropna(how='all'); print('active_dates',len(active),'coverage',np.mean([f.loc[x].notna().sum()/15 for x in f.index]),'turnover',np.nanmean((active-active.shift()).abs().mean(axis=1)))
# thirds H10
h=10; fr=px.shift(-h)/px-1; vals=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
v=pd.DataFrame(vals,columns=['d','ic']); print('thirds',v.groupby(pd.qcut(np.arange(len(v)),3,labels=False)).ic.mean().to_list())
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal'); out.to_csv('scripts/miner_2_20321227_dispersion_downside_reversal_signal.csv',index=False)
