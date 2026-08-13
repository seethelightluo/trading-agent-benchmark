src = open('scripts/miner_1_20340206_candidate_screen.py').read()
src = src.replace('vol60 = rets.rolling(60).std(ddof=0)\n', 'vol60 = rets.rolling(60).std(ddof=0)\nvol20 = rets.rolling(20).std(ddof=0)\n')
open('scripts/miner_1_20340206_candidate_screen.py','w').write(src)
print("patched")
