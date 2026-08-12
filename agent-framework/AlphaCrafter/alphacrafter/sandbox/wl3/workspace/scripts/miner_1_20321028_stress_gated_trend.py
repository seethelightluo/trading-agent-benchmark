import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(path):
 d=pd.read_csv(path,parse_dates=['date']).set_index('date').sort_index(); return d['close'].astype(float)
P=pd.concat({s:load('../persistent/stock_data/'+s+'.csv') for s in U},axis=1).sort_index().ffill(); r=P.pct_change()
dxy=load('../persistent/index_data/DXY.csv').reindex(P.index).ffill(); vix=load('../persistent/index_data/VIX.csv').reindex(P.index).ffill()
base=np.log(P/P.shift(20))/r.rolling(30,min_periods=20).std()
stress=((vix.pct_change(5)>0)&(dxy.pct_change(5)>0)).astype(float)
f=base.mul(1-0.55*stress,axis=0).rolling(3,min_periods=3).mean().shift(1)
fr=np.log(P.shift(-10)/P); rows=[]
for dt in f.index:
 a,b=f.loc[dt],fr.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic
print('dates',len(z),'avgN',z.n.mean(),'assets',len(U),'coverage',z.n.mean()/15,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for n in [120,252,756]:
 x=q.tail(n);print('recent',n,len(x),x.mean(),x.mean()/x.std(ddof=1))
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2032')]:
 x=q.loc[a:b]; print('regime',a,b,len(x),x.mean(),x.mean()/x.std(ddof=1))
for h in [1,3,5,10,20]:
 ff=np.log(P.shift(-h)/P); rr=[]
 for dt in f.index:
  a,b=f.loc[dt],ff.loc[dt];ok=a.notna()&b.notna()
  if ok.sum()>=8:rr.append(a[ok].corr(b[ok]))
 print('decay',h,np.nanmean(rr))
f.to_csv('scripts/miner_1_20321028_stress_gated_trend_signal.csv');z.to_csv('scripts/miner_1_20321028_stress_gated_trend_ic.csv')
print('signal_path scripts/miner_1_20321028_stress_gated_trend_signal.csv');print('ic_path scripts/miner_1_20321028_stress_gated_trend_ic.csv');print('library_corr null')
