import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
files=glob.glob('../persistent/stock_data/*.csv')
px={}
for f in files:
 s=os.path.basename(f)[:-4]; d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date'); px[s]=d['close']
close=pd.DataFrame(px).sort_index(); ret=close.pct_change()
# One idea: upside-capture / downside-risk balance, a defensive trend-quality signal.
# Factor at t uses returns through t; forward starts t+1.
up=ret.clip(lower=0).rolling(30,min_periods=20).mean()
down=(-ret.clip(upper=0)).rolling(30,min_periods=20).mean()
trend=ret.rolling(20,min_periods=15).mean()
factor=(trend/(down+1e-8))*(up/(up+down+1e-8))
# winsorize cross-section only for robustness
rows=[]
for h in [1,5,10]:
  ics=[]; nins=[]
  for dt in close.index:
    f=factor.loc[dt]; fr=close.shift(-h).loc[dt]/close.loc[dt]-1
    z=pd.concat([f,fr],axis=1).dropna();
    if len(z)>=8:
      ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); nins.append(len(z))
  a=np.array(ics); rows.append((h,len(a),np.nanmean(a),np.nanstd(a,ddof=1),np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12),np.mean(a>0),np.mean(nins)))
# turnover of ranks day-over-day (mean fraction changed ordering proxy)
ranks=factor.rank(axis=1,pct=True); turnover=ranks.diff().abs().mean(axis=1).mean()
print('dates',len(close),'instruments',close.shape[1],'period',close.index.min().date(),close.index.max().date())
print('horizon n IC ICIR hit avgN')
for x in rows: print('%d %d %.8f %.8f %.8f %.4f %.2f'%x)
print('coverage',factor.notna().sum().sum()/factor.size,'rank_turnover',turnover)
print('regimes')
for lo,hi in [('2020-01-01','2021-12-31'),('2022-01-01','2023-12-31'),('2024-01-01','2025-12-31'),('2026-01-01','2027-08-27')]:
 a=[]
 for dt in close.loc[lo:hi].index:
  z=pd.concat([factor.loc[dt],(close.shift(-1).loc[dt]/close.loc[dt]-1)],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(lo,hi,len(a),round(float(np.mean(a)),6) if a else None)
# signal artifact
factor.to_csv('scripts/miner_2_20270827_upside_capture_balance_signal.csv')
