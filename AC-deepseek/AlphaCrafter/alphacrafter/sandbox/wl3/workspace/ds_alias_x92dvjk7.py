import sys; sys.path.insert(0,'scripts')
src = open('scripts/miner_3_20291129_screen_batch28.py').read()
# print correlation part
i = src.find('rho')
print(src[src.find('library'):][:3000])