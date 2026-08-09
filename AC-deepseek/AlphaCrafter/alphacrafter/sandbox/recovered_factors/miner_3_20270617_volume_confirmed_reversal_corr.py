import pandas as pd,numpy as np,glob,os,json
from scipy.stats import spearmanr
D={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date') for f in glob.glob('../persistent/stock_data/*.csv')}; assets=sorted(D)
px=pd.concat({a:D[a]['close'] for a in assets},axis=1).sort_index(); vol=pd.concat({a:D[a]['volume'] for a in assets},axis=1).sort_index(); ret=px.pct_change()
vs=(vol/(vol.rolling(20,min_periods=10).median()+1e-12)).clip(upper=5); c=(-(ret.rolling(3,min_periods=3).sum())*np.log1p(vs.rolling(3,min_periods=3).mean())).shift(1)
F={'candidate':c,'rav':((px/px.shift(20)-1)/ret.rolling(20,min_periods=15).std()).shift(0),'volrev':(-(px/px.shift(5)-1)/ret.rolling(5,min_periods=4).std()),'rankacc':None,'eff':None,'rv':ret.rolling(20,min_periods=15).std(),'relvol':np.log(vol/(vol.rolling(20,min_periods=10).mean()+1e-12))}
F['rankacc']=-(ret.rolling(5).sum().rank(axis=1,pct=True)-ret.rolling(20).sum().rank(axis=1,pct=True))
F['eff']=ret.rolling(10).sum().abs()/(ret.abs().rolling(10).sum()+1e-12)
for k,v in F.items():
 if k=='candidate':continue
 z=pd.concat([c.stack(),v.stack()],axis=1).dropna(); print(k,round(abs(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic),6),len(z))
PY
python /dev/stdin <<'PY'
PY