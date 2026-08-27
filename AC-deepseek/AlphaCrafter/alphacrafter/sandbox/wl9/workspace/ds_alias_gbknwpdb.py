# Check revalidation was done recently - find last miner2 revalidation
import os
scripts = sorted([f for f in os.listdir('scripts') if f.startswith('miner2_') and 'revalidate' in f])
for s in scripts[-5:]:
    print(s)