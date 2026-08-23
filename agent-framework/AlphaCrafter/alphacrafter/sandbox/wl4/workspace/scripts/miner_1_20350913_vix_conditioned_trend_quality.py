import os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; macro='../persistent/index_data/VIX.csv'
px={}
for s in U:
    f=f'{base}/{s}.csv'
    if os.path.exists(f):
        d=pd.read_csv(f); d['date']=pd.to_datetime(d['date']); px[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(px).sort_index()
v=pd.read_csv(macro); v['date']=pd.to_datetime(v['date']); v=v.set_index('date')['close'].astype(float).reindex(p.index).ffill()
# factor at date d uses data strictly before d; stress scales trend quality down during high VIX
r=np.log(p).diff(); mom=p.shift(1).pct_change(20); vol=r.shift(1).rolling(20).std(); quality=mom/vol.replace(0,np.nan)
med=v.shift(1).rolling(60,min_periods=30).median(); stress=((v.shift(1)-med)/med.replace(0,np.nan)).clip(-1,1)
f=quality*(1-0.35*stress.fillna(0))
# forward 10 trading-day return, from d close to d+10 close; signal remains lagged
fr=p.shift(-10)/p-1
rows=[]; sigrows=[]
for d in p.index:
    a=f.loc[d]; b=fr.loc[d]; z=pd.concat([a,b],axis=1).dropna();
    if len(z)>=8:
        ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
        rows.append((d,ic,len(z)))
        for s,x in a.items(): sigrows.append((d,s,x))
res=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
res=res[(res.index>='2026-08-06')&(res.index<='2035-08-15')]
ic=res.ic.mean(); icir=ic/res.ic.std(ddof=1)*np.sqrt(252) if res.ic.std(ddof=1)>0 else np.nan
# rank turnover among consecutive available dates
ranks=f.rank(axis=1,pct=True); turn=(ranks.diff().abs().mean(axis=1)).dropna().mean()
coverage=f.notna().sum(axis=1).mean()/len(U)
out={'dates':len(res),'avg_n':res.n.mean(),'ic':ic,'icir':icir,'hit':(res.ic>0).mean(),'coverage':coverage,'turnover':turn}
for w in [120,260,520,780]:
 x=res.tail(w).ic; out[f'icir_{w}']=x.mean()/x.std(ddof=1)*np.sqrt(252) if len(x)>2 else np.nan
print(json.dumps(out,indent=2))
os.makedirs('scripts/artifacts',exist_ok=True)
pd.DataFrame(sigrows,columns=['date','symbol','signal']).to_csv('scripts/artifacts/miner_1_20350913_vix_conditioned_trend_quality_signal.csv',index=False)
res.reset_index().to_csv('scripts/artifacts/miner_1_20350913_vix_conditioned_trend_quality_ic.csv',index=False)
