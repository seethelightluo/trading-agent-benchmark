import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}; vol={}
for s in watch:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  q=d.set_index('date'); px[s]=q['close'].astype(float); vol[s]=q['volume'].astype(float)
px=pd.DataFrame(px).sort_index().ffill(); volume=pd.DataFrame(vol).reindex(px.index).ffill()
r=px.pct_change(); rv=r.rolling(30,min_periods=20).std().shift(1)*np.sqrt(30)
# Lagged 3-day reversal, strengthened by abnormal lagged volume and cross-asset stress.
rev=-px.pct_change(3).shift(1)/rv
volshock=(volume/volume.rolling(30,min_periods=20).median()).shift(1).clip(.5,3)
disp=r.std(axis=1,ddof=0); stress=(disp/disp.rolling(60,min_periods=40).median()).shift(1).clip(.5,2)
f=rev*volshock.mul(stress,axis=0)
fw_all={h:px.shift(-h)/px-1 for h in [1,5,10,20]}
rows=[]
for dt in f.index:
 a=f.loc[dt]; b=fw_all[10].loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8:
  z=a[ok].corr(b[ok],method='spearman')
  if pd.notna(z): rows.append((dt,z,ok.sum()))
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); m=ic.ic.mean(); sd=ic.ic.std(ddof=1)
print('dates',len(ic),'start',ic.index.min(),'end',ic.index.max(),'avg_n',ic.n.mean())
print('coverage',float(f.notna().mean().mean()),'mean_ic',m,'icir',m/sd*np.sqrt(252),'hit',float((ic.ic>0).mean()),'turnover',float(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for h in [1,5,10,20]:
 vals=[]
 for dt in f.index:
  a=f.loc[dt]; b=fw_all[h].loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8:
   z=a[ok].corr(b[ok],method='spearman')
   if pd.notna(z): vals.append(z)
 print('decay',h,'ic',float(np.mean(vals)),'dates',len(vals))
for label,sub in [('365d',ic.tail(252)),('180d',ic.tail(126)),('2032YTD',ic[ic.index>='2032-01-01'])]:
 if len(sub)>5: print(label,'n',len(sub),'ic',sub.ic.mean(),'icir',sub.ic.mean()/sub.ic.std(ddof=1)*np.sqrt(252))
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20320527_volume_confirmed_reversal_signal.csv',index=False)
ic.to_csv('scripts/miner_1_20320527_volume_confirmed_reversal_ic.csv')
