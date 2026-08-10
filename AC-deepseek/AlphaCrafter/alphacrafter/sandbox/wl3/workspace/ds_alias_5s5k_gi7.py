import re
src = open('scripts/factor_common.py').read()
i = src.find('def persist_factor')
print(src[i-30:i+1200])