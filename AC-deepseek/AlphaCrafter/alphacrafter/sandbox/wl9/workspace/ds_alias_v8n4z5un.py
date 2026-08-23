import json
print(open('/home/lxx/trade-agent-benchmark/AC-deepseek/AlphaCrafter/alphacrafter/sandbox/wl9/workspace/factor_ensemble.json').read())
print("===MEM===")
try:
    print(open('/home/lxx/trade-agent-benchmark/AC-deepseek/AlphaCrafter/alphacrafter/sandbox/wl9/workspace/memory.txt').read()[-3000:])
except Exception as e:
    print("mem err", e)