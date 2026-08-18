import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-02-04')
D={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}
p=pd.concat(D,axis=1).sort_index().loc[:cut]; lr=np.log(p).diff(); r10=lr.rolling(10).sum(); m=r10.mean(1); disp=r10.std(1)
# conditional signal: relative 10d reversal only when dispersion is above its trailing 60d median; zero otherwise
f=(-(r10.sub(m,axis=0))).where(disp>disp.rolling(60).median()).shift(1)
R=p.shift(-10)/p-1
A=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],R.loc[dt]],axis=1).dropna()
 if len(z)>=8:A.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
a=pd.DataFrame(A,columns=['date','ic','n']).set_index('date')
def st(x):return (x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1), (x.ic>0).mean(),len(x),x.n.mean())
print('dates/n',len(a),a.n.mean())
for k,x in [('all',a),('365',a.loc[cut-pd.Timedelta(365,'d'):]),('730',a.loc[cut-pd.Timedelta(730,'d'):]),('1095',a.loc[cut-pd.Timedelta(1095,'d'):]),('2028-30',a.loc['2028':'2030'])]:print(k,st(x))
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(1).mean())
