p='scripts/miner_2_20270909_revalidate_market_synchronization_60_20.py'
s=open(p).read()
s=s.replace("r=p.pct_change(); m=r.mean(axis=1); own=r.rolling(20,min_periods=15).std()", "r=p.pct_change(); m=r.mean(axis=1); own=r.rolling(20,min_periods=15).std(); dd0=p/p.rolling(60,min_periods=40).max()-1; breadth0=(dd0<-.05).mean(axis=1)")
open(p,'w').write(s)
print('fixed')
