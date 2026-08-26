import os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr

symbols=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in symbols:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).sort_values('date').set_index('date')
 px[s]=d['close'].replace(0,np.nan)
P=pd.DataFrame(px).sort_index()
r=P.pct_change()
# Contrarian return, scaled by realized volatility; all inputs at t
factor=-(P/P.shift(20)-1)/(r.rolling(20).std()*np.sqrt(20)+1e-12)
factor=factor.replace([np.inf,-np.inf],np.nan)
ics=[]; ns=[]; turnovers=[]
for i in range(20,len(P)-10):
 f=factor.iloc[i]; fr=P.iloc[i+10]/P.iloc[i]-1
 ok=f.notna()&fr.notna()
 if ok.sum()>=8:
  ics.append(spearmanr(f[ok],fr[ok]).statistic); ns.append(ok.sum())
# rank turnover over consecutive valid dates
ranks=factor.rank(axis=1,pct=True)
for i in range(21,len(P)):
 a=ranks.iloc[i-1]; b=ranks.iloc[i]; ok=a.notna()&b.notna()
 if ok.sum()>=8: turnovers.append(np.mean(np.abs(a[ok]-b[ok])))
ics=np.array(ics); print(json.dumps({'idea':'volatility-normalized 20d contrarian reversal','dates':len(ics),'avg_n':float(np.mean(ns)),'coverage':float(np.mean(ns)/15),'ic10':float(np.mean(ics)),'icir10':float(np.mean(ics)/(np.std(ics,ddof=1)+1e-12)*np.sqrt(252/10)),'hit':float(np.mean(ics>0)),'turnover_rank_abs':float(np.mean(turnovers))},indent=2))
# regimes
for a,b in [('2020','2023'),('2024','2027'),('2028','2031'),('2032','2035')]:
 mask=np.array([(str(P.index[i+10].date())[:4]>=a and str(P.index[i+10].date())[:4]<=b) for i in range(20,len(P)-10) if factor.iloc[i].notna().sum()>=8])
 x=ics[mask] if len(mask)==len(ics) else ics # dates align generally
 print(a,b, round(float(np.mean(x)),5),round(float(np.mean(x)/(np.std(x,ddof=1)+1e-12)*np.sqrt(252/10)),4),len(x))
