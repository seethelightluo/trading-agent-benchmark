p='scripts/miner_3_20350330_compression_trend.py'
s=open(p).read();s=s.replace("raw=P.pct_change(20)*(r.rolling(60).std()/r.rolling(10).std()).clip(.25,4)","raw=P.pct_change(20)/r.rolling(20).std()")
s=s.replace('compression_trend','volscaled_momentum20')
open('scripts/miner_3_20350330_volscaled_momentum20.py','w').write(s)
PY
python scripts/miner_3_20350330_volscaled_momentum20.py | head - 12