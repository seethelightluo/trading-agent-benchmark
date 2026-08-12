import pathlib
lines = pathlib.Path('memory.txt').read_text().splitlines()
# find feedback to Miner / miner lines
for l in lines:
    if 'Miner' in l or 'miner' in l.lower():
        print(l[:300])
        print('===')
