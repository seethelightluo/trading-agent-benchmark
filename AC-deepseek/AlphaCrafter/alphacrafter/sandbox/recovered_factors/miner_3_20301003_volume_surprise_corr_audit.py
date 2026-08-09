import pandas as pd,numpy as np,glob,os,json
from scipy.stats import spearmanr
D={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date') for f in glob.glob('../persistent/stock_data/*.csv')}
p=pd.DataFrame({k:v.close for k,v in D.items()}).sort_index().astype(float);v=pd.DataFrame({k:v.volume for k,v in D.items()}).reindex(p.index).astype(float);r=p.pct_change()
vr=(v.shift(1).rolling(5,min_periods=3).mean()/v.shift(1).rolling(60,min_periods=30).mean()).clip(.25,4)
cand=vr.rank(axis=1,pct=True)
lib={
'relative_volume_participation_20':np.log(v.shift(1)/v.shift(1).rolling(20,min_periods=10).mean()),
'risk_adjusted_trend_20':(p.shift(1)/p.shift(21)-1)/r.rolling(20,min_periods=15).std().shift(1),
'ret20':p.shift(1)/p.shift(21)-1,
'inverse_vol20':-r.rolling(20,min_periods=15).std().shift(1),
'volume_ratio_5_60':vr
}
mx=0;best=''
for n,s in lib.items():
 z=pd.concat([cand.stack().rename('a'),s.stack().rename('b')],axis=1).dropna();q=abs(spearmanr(z.a,z.b).statistic);print(n,'rho',round(q,6),'n',len(z));
 if q>mx:mx=q;best=n
print('max_abs_library_correlation',round(mx,6),best)
print('coverage',round(cand.notna().sum().sum()/cand.size,6),'dates',len(p),'instruments',len(p.columns))
# list effective files for audit inventory
print('effective_file_count',sum(json.load(open(f)).get('validation',{}).get('status')=='EFFECTIVE' for f in glob.glob('factors/*.json') if not f.endswith('.bak')))
