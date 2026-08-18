lines = open('scripts/miner_3_20280518_screen_batch.py').read().splitlines()
for i in range(59, 160):
    print(f"{i+1}: {lines[i]}")