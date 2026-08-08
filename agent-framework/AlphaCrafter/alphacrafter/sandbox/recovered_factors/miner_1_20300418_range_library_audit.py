exec(open('scripts/miner_1_20300418_range_position_reversal.py').read().split("print('assets'")[0])
# pooled Spearman correlation against reconstructed admitted signal families; complete evidence only on common cells
from scipy.stats import spearmanr
libs={
'risk_adjusted_trend_20':p.pct_change(20)/r.rolling(20).std(),
'volnorm_reversal_5':-p.pct_change(5)/r.rolling(5).std(),
'ravmom_20':p.pct_change(20)/r.rolling(20).std(),
'inverse_range_position':(p-lo)/(hi-lo),
'trend_consistency':(p.pct_change(20)/r.rolling(20).std())*((r>0).rolling(20).mean()-.5),
'curvature':(r.rolling(20).sum()/(r.rolling(20).std()*np.sqrt(20)))-(r.rolling(60).sum()/(r.rolling(60).std()*np.sqrt(60)))
}
for k,v in libs.items():
 z=pd.concat([f.stack(),v.shift(1).stack()],axis=1).dropna()
 rho=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
 print('LIB',k,'rho %.6f cells %d'%(rho,len(z)))
