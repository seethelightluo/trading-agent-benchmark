src = open('strategy.py').read()
i = src.find('def _load_ensemble')
print(src[i:i+1400])