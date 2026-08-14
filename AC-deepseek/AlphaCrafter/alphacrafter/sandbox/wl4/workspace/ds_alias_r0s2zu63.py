lines = open('memory.txt').read().splitlines()
for i, l in enumerate(lines):
    if 'miner' in l.lower() and ('batch' in l.lower() or 'factor' in l.lower() or 'explore' in l.lower()):
        print(i, l[:200])
        print('~~~')
