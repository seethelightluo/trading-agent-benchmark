import os
os.chdir('/home/lxx/trade-agent-benchmark/AC-deepseek/AlphaCrafter/alphacrafter/sandbox/wl9/workspace')
import sys
# run sweep J
sys.path.insert(0,'.')
exec(open('scripts/miner2_20260819_sweepJ_timeprice.py').read())