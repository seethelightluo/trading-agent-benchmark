# patch ambiguous date grouping
p='scripts/miner_3_20281116_candle_pressure.py'
s=open(p).read().replace("q['date']=q.index;rr.append(q)","q=q.reset_index(names='date');rr.append(q)")
open(p,'w').write(s)
