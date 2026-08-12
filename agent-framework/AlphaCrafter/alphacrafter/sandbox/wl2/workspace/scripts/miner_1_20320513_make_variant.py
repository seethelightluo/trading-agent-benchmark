# Variant runner: defensive-relative 3-day shock reversal
p='scripts/miner_1_20320513_defensive_relative_reversal.py'
s=open(p).read().replace('r5=px.pct_change(5);','r5=px.pct_change(3);').replace("'defensive_relative_reversal_signal.csv'","'defensive_relative3_reversal_signal.csv'")
open('scripts/miner_1_20320513_defensive_relative3_reversal.py','w').write(s)
