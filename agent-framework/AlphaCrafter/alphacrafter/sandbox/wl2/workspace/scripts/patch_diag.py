import pandas as pd,numpy as np
p='scripts/miner_3_20321028_vix_breadth_capitulation.py';s=open(p).read();s=s.replace("q=pd.concat([f.loc[d],R.shift(-1).loc[d]],axis=1).dropna()","q=pd.concat([f.loc[d],R.shift(-1).loc[d]],axis=1).dropna();\n if len(q)<8 and active.loc[d]>0: print('diag',d,len(q),int(f.loc[d].notna().sum()),int(R.shift(-1).loc[d].notna().sum()))")
open(p,'w').write(s)
