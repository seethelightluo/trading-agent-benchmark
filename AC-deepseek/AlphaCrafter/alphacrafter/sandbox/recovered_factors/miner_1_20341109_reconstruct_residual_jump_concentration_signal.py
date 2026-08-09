"""Reconstruct and serialize the admitted residual jump-concentration signal.
Uses the original producer setup/cutoff and exact persisted expression."""
import pickle
src=open('scripts/miner_1_20310403_residual_jump_concentration_expansion_20_60d.py',encoding='utf8').read()
prefix=src.split('# Candidate: idiosyncratic return concentration.')[0]
exec(prefix,globals())
def jump_share(x,w,n):
    def calc(z):
        a=np.abs(np.asarray(z,float)); k=max(1,int(np.ceil(.20*len(a))))
        return np.partition(a,-k)[-k:].sum()/a.sum() if a.sum()>0 else np.nan
    return x.rolling(w,min_periods=n).apply(calc,raw=True)
f=pd.DataFrame({a:jump_share(e[a],20,14)-jump_share(e[a],60,42) for a in A})
factor_id='miner_1_residual_jump_concentration_expansion_20_60d'
out='scripts/'+factor_id+'_signal.pkl'
with open(out,'wb') as h:
    pickle.dump({'factor_id':factor_id,'producer':'miner_1_20310403_residual_jump_concentration_expansion_20_60d.py','end':str(END.date()),'symbols':A,'signal':f},h)
print('SERIALIZED',out,'rows',len(f),'cols',len(f.columns),'start',f.index.min().date(),'end',f.index.max().date(),'coverage',round(float(f.notna().mean().mean()),6),'inherited_library_signals',len(lib))
print('FACTOR_EXPRESSION jump_share(abs(residual_return),20,min_periods=14)-jump_share(abs(residual_return),60,min_periods=42); jump_share=sum(largest ceil(20% daily moves))/sum(all daily moves); residual_return=r_i-equal_weight_return')
