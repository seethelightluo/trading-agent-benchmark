import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,2500)
 if d is None or len(d)<100:d=get_index_daily_data(s,2500)
 D[s]=d
p=pd.concat({s:d.set_index('date')['close'] for s,d in D.items() if d is not None},axis=1).sort_index(); r=np.log(p).diff(); ret=r.rolling(10).sum(); vol=r.rolling(20).std()*np.sqrt(252); rows=[]
# dispersion history precomputed
csdisp=ret.std(axis=1); med=csdisp.rolling(120,min_periods=20).median()
for i in range(120,len(p)-10):
 f=(ret.iloc[i]-ret.iloc[i].median())/(vol.iloc[i]+1e-8); gate=1.0 if csdisp.iloc[i]>=med.iloc[i] else .35; f*=gate; fw=np.log(p.iloc[i+10]/p.iloc[i]); z=pd.concat([f.rename('f'),fw.rename('fw')],axis=1).dropna()
 if len(z)>=8:rows.append((p.index[i],z.f.corr(z.fw),len(z),gate))
o=pd.DataFrame(rows,columns=['date','ic','n','gate']).set_index('date'); print('dates',len(o),'mean_n',o.n.mean(),'coverage',o.n.mean()/15,'meanIC',o.ic.mean(),'ICIR',o.ic.mean()/o.ic.std(),'hit',(o.ic>0).mean(),'turnover_proxy',o.ic.diff().abs().mean());
for a,b in [('2020','2025-12-31'),('2026','2029-12-31'),('2030','2032-12-31'),('2033','2034-12-31')]:
 x=o.loc[a:b,'ic'];print(a,b,len(x),x.mean(),x.mean()/x.std() if x.std()>0 else np.nan)
o.to_csv('../persistent/miner_3_20340901_residual_momentum_ic.csv')
