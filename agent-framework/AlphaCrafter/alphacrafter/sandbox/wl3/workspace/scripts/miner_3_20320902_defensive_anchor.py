import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).ffill()
# Defensive-anchor spread: lagged relative momentum, tilted toward assets
# with strength when defensive assets outperform equity assets.
defens=['XAU','US10Y','CN10Y']; eq=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX']
r=np.log(P/P.shift(20)); rel=r.sub(r.median(axis=1),axis=0)
anchor=r[defens].mean(axis=1)-r[eq].mean(axis=1)
# rolling cross-sectional/regime normalization uses completed data only
az=(anchor-anchor.rolling(252,min_periods=120).mean())/anchor.rolling(252,min_periods=120).std()
# smooth bounded macro tilt; preserves ranking direction but rewards defensive regime alignment
vol=P.pct_change().rolling(20,min_periods=15).std()*np.sqrt(252)
base=rel/vol.replace(0,np.nan)
f=base.mul((1+0.35*np.tanh(az)).clip(.65,1.35),axis=0).rolling(5,min_periods=5).mean().shift(1)
fr=np.log(P.shift(-10)/P); rows=[]
for dt in f.index:
 a,b=f.loc[dt],fr.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic
print('dates',len(z),'avgN',z.n.mean(),'assets',len(U),'coverage',z.n.mean()/len(U),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for n in [120,252,756]:
 x=q.tail(n);print('recent',n,len(x),x.mean(),x.mean()/x.std(ddof=1))
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2032')]:
 x=q.loc[a:b]; print(a,b,len(x),x.mean(),x.mean()/x.std(ddof=1))
f.to_csv('scripts/miner_3_20320902_defensive_anchor_signal.csv');z.to_csv('scripts/miner_3_20320902_defensive_anchor_ic.csv')
# print signal artifact provenance
print('signal_path scripts/miner_3_20320902_defensive_anchor_signal.csv')
print('ic_path scripts/miner_3_20320902_defensive_anchor_ic.csv')
print('library_corr null (requires deterministic post-Miner recomputation)')
