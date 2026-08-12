import glob, os
scripts = sorted(glob.glob('scripts/miner*'))
print('\n'.join(scripts[-15:]))
print('---')
# Look at a recent miner script for the data-loading pattern
if scripts:
    print(open(scripts[-1]).read()[:4000])
