import pandas as pd,numpy as np
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2033-06-12')
D={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').loc[:cut]
 D[a]=d.close.pct_change(20).shift(1); D[a+'_c']=d.close
R=pd.DataFrame({a:D[a] for a in assets}); med=R.median(axis=1); F=R.sub(med,axis=0)
ics={h:[] for h in [1,5,10,20,30]}; ns=[]
for h in ics:
 fw=pd.DataFrame({a:D[a+'_c'].shift(-h)/D[a+'_c']-1 for a in assets})
 for dt in F.index:
  z=pd.concat([F.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q): ics[h].append(q)
for h,x in ics.items():
 x=np.array(x); print('H',h,'n',len(x),'IC %.6f ICIR %.6f hit %.4f'%(x.mean(),x.mean()/x.std(ddof=1),np.mean(x>0)))
x=np.array(ics[10]); print('thirds',*[round(y.mean(),6) for y in np.array_split(x,3)])
print('coverage',F.notna().mean().mean(),'dates',len(x),'assets',len(assets))
out=[]
for dt in F.index:
 for a in assets: out.append({'date':dt.date(),'asset':a,'signal':F.loc[dt,a]})
pd.DataFrame(out).to_csv('scripts/miner_2_20330613_relative_momentum_signal.csv',index=False)
