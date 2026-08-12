import sys; sys.path.insert(0,'scripts')
src = open('scripts/miner_3_20291129_screen_batch28.py').read()
# find the tail where candidates get evaluated and correlations computed
i = src.find("def f_")
print(src[i:i+4000])