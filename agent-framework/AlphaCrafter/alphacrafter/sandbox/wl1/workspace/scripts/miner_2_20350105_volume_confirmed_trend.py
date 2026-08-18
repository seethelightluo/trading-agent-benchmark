import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
d={s:load(s) for s in U}; p=pd.concat({s:x.close.astype(float) for s,x in d.items()},axis=1).sort_index().ffill(); r=p.pct_change()
# Candidate: trend persistence with volume/liquidity confirmation. Volume surprise is causal and cross-sectionally ranked.
vol=pd.concat({s:x.volume.astype(float) for s,x in d.items()},axis=1).reindex(p.index).ffill()
mom=r.rolling(40).sum(); risk=r.rolling(20).std()*np.sqrt(20); volume_ratio=vol.rolling(10).mean()/vol.rolling(60).mean().replace(0,np.nan)
# damp unstable trends; volume confirmation multiplier bounded to avoid crypto dominance
confirm=(0.75+0.5*volume_ratio.clip(0.5,1.5)).clip(0.5,1.5)
f=(mom/risk.replace(0,np.nan)*confirm).shift(1)
rows=[]
for i in range(len(p)-10):
 z=pd.concat([f.iloc[i],r.iloc[i+1:i+11].sum()],axis=1).dropna()
 if len(z)>=8: rows.append((p.index[i],len(z),z.iloc[:,0].corr(z.iloc[:,1])))
df=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); ic=df.ic.dropna()
print('candidate=volume_confirmed_trend_40d; dates',len(df),'avgN',df.n.mean(),'coverage',df.n.mean()/15,'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2029-12-31'),('2030-01-01','2034-12-31'),('2034-01-01','2035-01-05')]:
 x=ic.loc[a:b]; print(a,b,'dates',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1) if len(x)>1 else np.nan)
rank=f.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean(axis=1).dropna().mean(),'recentIC',ic.tail(260).mean(),'recentICIR',ic.tail(260).mean()/ic.tail(260).std(ddof=1))
f.to_csv('scripts/miner_2_20350105_volume_confirmed_trend_signal.csv')
