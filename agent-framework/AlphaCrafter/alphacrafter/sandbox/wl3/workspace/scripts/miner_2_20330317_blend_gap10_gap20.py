import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items() for _ in [0]}).ffill()
# Blend two interpretable medium-term reversal horizons, each excluding recent sessions.
a=-np.log(P.shift(10)/P.shift(70)); b=-np.log(P.shift(20)/P.shift(80))
a=a.sub(a.median(axis=1),axis=0); b=b.sub(b.median(axis=1),axis=0)
f=(0.5*a+0.5*b).shift(1)
fr=np.log(P.shift(-10)/P); rows=[]
for dt in f.index:
 x,y=f.loc[dt],fr.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((dt,x[ok].corr(y[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic
print('dates',len(z),'avgN',z.n.mean(),'assets',len(U),'coverage',z.n.mean()/15,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for n in [120,252,756]:
 x=q.tail(n); print('recent',n,len(x),x.mean(),x.mean()/x.std(ddof=1))
for a0,b0 in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2032'),('2033','2033')]:
 x=q.loc[a0:b0]; print('regime',a0,b0,len(x),x.mean(),x.mean()/x.std(ddof=1) if len(x)>1 else np.nan)
f.to_csv('scripts/miner_2_20330317_blend_gap10_gap20_signal.csv'); z.to_csv('scripts/miner_2_20330317_blend_gap10_gap20_ic.csv')
print('signal_path scripts/miner_2_20330317_blend_gap10_gap20_signal.csv'); print('ic_path scripts/miner_2_20330317_blend_gap10_gap20_ic.csv'); print('library_corr unavailable')
