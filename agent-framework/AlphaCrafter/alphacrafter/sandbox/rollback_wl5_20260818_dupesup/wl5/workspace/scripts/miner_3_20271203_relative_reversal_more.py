import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
for L,H in [(20,20),(10,5),(5,10)]:
 P=pd.DataFrame({s:D[s].close.pct_change(L) for s in U});F=-P.sub(P.median(axis=1),axis=0);Y=pd.DataFrame({s:D[s].close.shift(-H)/D[s].close-1 for s in U}); out=[]
 for d in F.index:
  g=pd.DataFrame({'f':F.loc[d],'y':Y.loc[d]}).dropna()
  if d>=pd.Timestamp('2020-01-01') and len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:out.append(spearmanr(g.f,g.y).statistic)
 z=np.array(out);print(L,H,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6),round((z>0).mean(),4))
