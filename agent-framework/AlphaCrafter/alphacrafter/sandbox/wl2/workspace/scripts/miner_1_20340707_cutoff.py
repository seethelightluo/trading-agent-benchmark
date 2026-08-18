import os
p='scripts/miner_1_20340707_short_reversal.py'
s=open(p).read(); s=s.replace("prices=pd.DataFrame(px).sort_index()", "prices=pd.DataFrame(px).sort_index(); prices=prices.loc[prices.index <= '2034-07-07']")
open(p,'w').write(s)
p='scripts/miner_1_20340707_stress_reversal.py'
s=open(p).read(); s=s.replace("p=pd.DataFrame(px).sort_index();", "p=pd.DataFrame(px).sort_index(); p=p.loc[p.index <= '2034-07-07'];")
open(p,'w').write(s)
PY
python scripts/miner_1_20340707_short_reversal.py && python scripts/miner_1_20340707_stress_reversal.py
PY