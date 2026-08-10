src = open('scripts/factor_common.py').read()
i = src.find('def save_signal_artifact')
print(src[i:i+1800])