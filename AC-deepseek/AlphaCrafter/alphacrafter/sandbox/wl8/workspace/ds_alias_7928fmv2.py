import os, glob
print("ROOT JSON files in factors/:")
for f in sorted(glob.glob('factors/*.json')):
    print(' ', f)
print()
print("ROOT .bak count:", len(glob.glob('factors/*.bak')))
print("evicted count:", len(glob.glob('factors/evicted/*.json')) - len(glob.glob('factors/evicted/*.reason.json')))
print("rejected count:", len(glob.glob('factors/rejected/*.json')) - len(glob.glob('factors/rejected/*.reason.json')))