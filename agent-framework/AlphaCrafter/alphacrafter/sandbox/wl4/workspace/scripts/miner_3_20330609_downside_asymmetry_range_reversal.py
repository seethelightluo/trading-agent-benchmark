import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; sig={}; px={}
for a in assets:
 f=f'{base}/{a}.csv'
 if not os.path.exists(f): continue
 d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date'); c=d.close.astype(float); r=c.pct_change()
 mom=c.pct_change(20); vol=r.rolling(60,min_periods=30).std()*np.sqrt(20)
 hi=c.rolling(60,min_periods=40).max(); lo=c.rolling(60,min_periods=40).min()
 loc=((c-lo)/(hi-lo).replace(0,np.nan)).rolling(10,min_periods=5).mean()
 down=r.clip(upper=0).rolling(40,min_periods=20).std(); asym=(down/vol.replace(0,np.nan)).clip(0.25,2.5)
 # Mean reversion is amplified when recent downside risk is unusually large, with range-location conditioning.
 sig[a]=(-mom/vol*(0.5+loc)*asym).shift(1); px[a]=c
F=pd.DataFrame(sig); P=pd.DataFrame(px); rows=[]
for dt in F.index:
 z=pd.concat([F.loc[dt],P.shift(-10).loc[dt]/P.loc[dt]-1],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('candidate downside_asymmetry_range_reversal_10d')
print('dates',len(r),'avgN',round(r.n.mean(),3),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),6))
for n in [260,520,780]:
 q=s.tail(min(n,len(s))); print('recent',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),6))
print('coverage',round(F.notna().sum(axis=1).mean()/len(assets),6),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
os.makedirs('scripts/artifacts',exist_ok=True); r.to_csv('scripts/artifacts/miner_3_20330609_downside_asymmetry_range_reversal_ic.csv',index=False); F.to_csv('scripts/artifacts/miner_3_20330609_downside_asymmetry_range_reversal_signal.csv')
