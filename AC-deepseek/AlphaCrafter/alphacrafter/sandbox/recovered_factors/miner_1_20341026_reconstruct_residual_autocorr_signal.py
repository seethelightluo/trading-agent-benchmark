"""Reconstruct and serialize one admitted factor's historical signal matrix.
The producer's setup/library is reused verbatim; its factor expression is reimplemented
from the JSON definition after setup, on its original visible-data cutoff.
"""
import os, pickle
src=open('scripts/miner_1_20310306_residual_return_autocorrelation_expansion_20_60d.py',encoding='utf8').read()
prefix=src.split('# Candidate: inverse of the independently specified persistence transition.')[0]
exec(prefix,globals())
def ac1(x,w,n):
    return x.rolling(w,min_periods=n).corr(x.shift(1))
f=pd.DataFrame({a:ac1(e[a],20,14)-ac1(e[a],60,42) for a in A})
factor_id='miner_1_residual_return_autocorrelation_expansion_20_60d'
out='scripts/'+factor_id+'_signal.pkl'
with open(out,'wb') as h:
    pickle.dump({'factor_id':factor_id,'producer':'miner_1_20310306_residual_return_autocorrelation_expansion_20_60d.py','end':str(END.date()),'symbols':A,'signal':f},h)
print('SERIALIZED',out,'rows',len(f),'cols',len(f.columns),'start',f.index.min().date(),'end',f.index.max().date(),'coverage',round(float(f.notna().mean().mean()),6),'inherited_library_signals',len(lib))
print('FACTOR_EXPRESSION acorr_1(residual_return,20,min_periods=14)-acorr_1(residual_return,60,min_periods=42); residual_return=r_i-equal_weight_return')
