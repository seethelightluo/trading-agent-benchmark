import json,runpy,io,contextlib
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from alphacrafter.sim import utils
# execute the canonical serial-dependence script; its cand is that factor signal
with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()): ns=runpy.run_path('scripts/miner_1_20300822_serial_dependence_library_audit.py')
g=ns['cand']; assets=utils.get_account_dict()['watch_list']; P=ns['P']; r=P.pct_change()
f=(-np.sign(r.sub(r.median(axis=1),axis=0)).rolling(60,min_periods=45).mean()).shift(1);f=f.sub(f.median(axis=1),axis=0)
q=pd.concat([f.stack().rename('candidate'),g.stack().rename('library')],axis=1).dropna(); rho=spearmanr(q.candidate,q.library).statistic
path='scripts/miner_2_20310109_directional_consistency_audit_cache_corrected.json';out=json.load(open(path));out['records']['inverse_peer_relative_serial_dependence_20']={'rho':float(rho),'cells':len(q),'source':'miner_1_20300822_serial_dependence_library_audit.py'};out['failures'].pop('inverse_peer_relative_serial_dependence_20',None);json.dump(out,open(path,'w'),indent=2);print('AUDIT rho',rho,'cells',len(q))
