import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; end=pd.Timestamp('2027-06-04'); px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).sort_values('date').set_index('date'); px[s]=d['close'].astype(float)
p=pd.DataFrame(px).sort_index().loc[:end].ffill(); r=p.pct_change(); down=r.where(r<0,0).rolling(15,min_periods=10).std(); f=(p.shift(1)/p.shift(16)-1)/down.shift(1)
res=[]
for dt in f.index:
 x=f.loc[dt]
 for h in [1,5,10]:
  y=p.pct_change(h).shift(-h).loc[dt]; z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8: res.append((dt,h,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
z=pd.DataFrame(res,columns=['date','h','n','ic']); z['date']=pd.to_datetime(z.date)
print('factor=downside_sortino_15d dates',z[z.h==1].date.nunique(),'rows',len(z[z.h==1]),'avg_n',z[z.h==1].n.mean())
for h in [1,5,10]:
 a=z[z.h==h].ic.dropna(); print(h,'IC %.5f ICIR %.5f hit %.3f'%(a.mean(),a.mean()/a.std(ddof=1), (a>0).mean()))
valid=f.notna().sum(axis=1); print('coverage',valid.mean()/15,'median valid',valid.median()); ranks=f.rank(axis=1,pct=True); print('turnover',ranks.diff().abs().mean(axis=1).dropna().mean())
for h in [1,5,10]:
 a=z[z.h==h]; print('regimes',h,[(str(yr),round(g.ic.mean(),5),len(g)) for yr,g in a.groupby(a.date.dt.year)])
f.reset_index().to_csv('scripts/miner_2_20270604_downside_sortino_15d_signal.csv',index=False)
