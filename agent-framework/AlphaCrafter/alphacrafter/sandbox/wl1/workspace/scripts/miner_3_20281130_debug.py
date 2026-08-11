# debug append
exec(open('scripts/miner_3_20281130_equal_residual_momentum.py').read().replace("r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');", "print('px',px.shape,'ret',ret.notna().sum().to_dict(),'sig',sig.notna().sum().to_dict(),'fwd',fwd.notna().sum().to_dict()); r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');"))
