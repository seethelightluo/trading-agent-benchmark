"""One candidate: inverse residual return-autocorrelation contraction (20d versus 60d)."""
import json,numpy as np,pandas as pd
src=open('scripts/miner_1_20310220_residual_return_autocorrelation_contraction_60_20d.py',encoding='utf8').read()
# The prior candidate's setup, data visibility, library reconstruction, and report framework.
src=src.replace('"""One candidate: residual return-autocorrelation contraction (60d versus 20d)."""','"""setup"""')
src=src.replace("# Candidate: each asset's own residual-return persistence, measured as lag-one\n# autocorrelation after removing the equal-weight cross-asset market return.\n# The signal is the structural (60d) persistence minus recent (20d) persistence:\n# positive readings identify assets whose serial dependence has recently weakened.\ndef ac1(x,w,n):\n    return x.rolling(w,min_periods=n).corr(x.shift(1))\nf=pd.DataFrame({a:ac1(e[a],60,42)-ac1(e[a],20,14) for a in A})", "# Candidate: inverse of the independently specified persistence transition.\n# After removing equal-weight market returns, lag-one residual autocorrelation\n# is estimated over 20 and 60 days.  Signal=recent persistence minus\n# structural persistence: positive readings mean serial dependence strengthened.\ndef ac1(x,w,n):\n    return x.rolling(w,min_periods=n).corr(x.shift(1))\nf=pd.DataFrame({a:ac1(e[a],20,14)-ac1(e[a],60,42) for a in A})")
src=src.replace('residual_return_autocorrelation_contraction_60_20d','residual_return_autocorrelation_expansion_20_60d')
open('scripts/miner_1_20310306_residual_return_autocorrelation_expansion_20_60d.py','w',encoding='utf8').write(src)
print('wrote candidate script')
