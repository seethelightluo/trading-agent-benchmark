import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).ffill(); lr=np.log(P).diff(); ret20=np.log(P/P.shift(20)); path=lr.abs().rolling(20,min_periods=15).sum(); vol=lr.rolling(20,min_periods=15).std()
quality=ret20/(path+1e-12)/(vol*np.sqrt(20)+1e-12)
disp=ret20.std(axis=1).rolling(20,min_periods=15).mean(); base=disp.rolling(252,min_periods=100).median(); state=(disp/(base+1e-12)).clip(.5,2)
f=(quality*state.shift(1).values[:,None]).shift(1)
fr=np.log(P.shift(-10)/P); rows=[]
for dt in f.index:
 a,b=f.loc[dt],fr.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic
print('dates',len(z),'avgN',z.n.mean(),'assets',len(U),'coverage',z.n.mean()/15,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for n in [120,252,756]:
 x=q.tail(n); print('recent',n,len(x),x.mean(),x.mean()/x.std(ddof=1))
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2032'),('2033','2033')]:
 x=q.loc[a:b]; print(a,b,len(x),x.mean(),x.mean()/x.std(ddof=1) if len(x)>1 else np.nan)
f.to_csv('scripts/miner_2_20330217_dispersion_quality_signal.csv'); z.to_csv('scripts/miner_2_20330217_dispersion_quality_ic.csv')
print('signal_path scripts/miner_2_20330217_dispersion_quality_signal.csv'); print('ic_path scripts/miner_2_20330217_dispersion_quality_ic.csv'); print('max_abs_library_correlation unavailable')
