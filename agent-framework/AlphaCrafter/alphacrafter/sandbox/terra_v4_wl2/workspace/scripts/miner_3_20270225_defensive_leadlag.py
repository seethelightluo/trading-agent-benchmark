import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-02-25')
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date <= @CUT').sort_values('date').set_index('date') for a in U}
px=pd.DataFrame({a:p[a].close for a in U}).sort_index(); r=px.pct_change()
deftrend=(r['XAU'].rolling(5).sum()-r['US10Y'].rolling(5).sum()).shift(1)
resid=r.rolling(3).sum().sub(r.rolling(3).sum().median(axis=1),axis=0)
f=(-resid).mul(np.tanh(deftrend/0.05),axis=0); fr=px.shift(-5)/px-1
rows=[]
for d in f.index:
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
df=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('dates',len(df),'avg_n',df.n.mean(),'coverage',df.n.mean()/15,'IC',df.ic.mean(),'ICIR',df.ic.mean()/df.ic.std(ddof=1),'hit',(df.ic>0).mean())
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027-02-25')]:
 q=df.loc[lo:hi].ic; print(lo,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
f.reset_index().melt(id_vars='date',var_name='asset',value_name='signal').to_csv('../persistent/factor_signals_miner_3_20270225_defensive_leadlag.csv',index=False)
