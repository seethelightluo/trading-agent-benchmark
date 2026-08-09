import json
p='scripts/miner_3_20310612_dxy_vix_state_impulse_exposure_5v40v20obs.py'
s=open(p).read()
s=s.replace("F=beta.mul(-dxr.rolling(5,min_periods=5).sum()*state,axis=0).loc[:END]", "F=beta.mul(-dxr.rolling(5,min_periods=5).sum(),axis=0).where(state.astype(bool), np.nan).loc[:END]")
open(p,'w').write(s)
