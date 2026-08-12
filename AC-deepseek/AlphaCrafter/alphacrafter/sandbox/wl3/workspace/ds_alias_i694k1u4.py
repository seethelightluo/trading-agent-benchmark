import sys; sys.path.insert(0,'scripts')
src = open('scripts/miner_3_20291129_screen_batch28.py').read()
i = src.find('max_lib_corr')
print(src[i:i+3000])