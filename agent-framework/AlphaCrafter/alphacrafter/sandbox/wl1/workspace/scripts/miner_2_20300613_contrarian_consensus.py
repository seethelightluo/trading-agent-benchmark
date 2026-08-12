import runpy
# reuse exact validated construction, then invert for contrarian interpretation
p='scripts/miner_2_20300613_consensus_trend.py'
s=open(p).read().replace("f=(0.45*ret20/(vol+0.01)+0.35*ret60/(vol+0.01)+0.20*ret120/(vol+0.01))", "f=-(0.45*ret20/(vol+0.01)+0.35*ret60/(vol+0.01)+0.20*ret120/(vol+0.01))").replace("miner_2_20300613_consensus_trend_signal.csv", "miner_2_20300613_contrarian_consensus_signal.csv")
exec(compile(s,'<contrarian>','exec'))
