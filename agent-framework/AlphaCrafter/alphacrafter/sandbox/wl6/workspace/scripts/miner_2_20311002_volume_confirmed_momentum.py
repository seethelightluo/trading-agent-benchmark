import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}; vol={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 px[s]=d.close.astype(float); vol[s]=d.volume.astype(float)
P=pd.DataFrame(px).sort_index().loc[:'2031-10-01']; V=pd.DataFrame(vol).reindex(P.index)
R=P.pct_change(); ret60=P/P.shift(60)-1
# Momentum strengthened by persistent volume participation, cross-sectionally ranked components.
vr=V.rolling(20,min_periods=10).mean()/V.rolling(120,min_periods=60).mean()-1
f=ret60.rank(axis=1,pct=True)* (1+vr.rank(axis=1,pct=True).fillna(.5))
def evalh(h):
 fr=P.shift(-h)/P-1; out=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c): out.append((dt,c,len(z)))
 q=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); ic=q.ic.mean(); sd=q.ic.std(ddof=1)
 print('H',h,'valid_dates',len(q),'avg_n',round(q.n.mean(),3),'IC',round(ic,8),'ICIR_daily',round(ic/sd,8),'ICIR_ann',round(ic/sd*np.sqrt(252),8),'hit',round((q.ic>0).mean(),4),'years',q.groupby(q.index.year).ic.mean().round(5).to_dict())
for h in [5,10,20]: evalh(h)
print('candidate=volume_confirmed_momentum dates=',len(P),'instruments=',len(U),'through=',P.index.max().date())
print('coverage',round(f.notna().sum(axis=1).mean()/15,6),'turnover_proxy',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
