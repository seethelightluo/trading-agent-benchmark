import os
print([f for f in os.listdir('scripts') if '2027' in f and ('miner_2' in f or 'miner_3' in f)][-12:])