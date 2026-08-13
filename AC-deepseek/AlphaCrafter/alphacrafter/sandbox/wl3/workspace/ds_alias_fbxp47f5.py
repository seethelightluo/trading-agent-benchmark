import re
mem = open('memory.txt').read()
# find miner-related summary sections
idx = [m.start() for m in re.finditer(r'(?i)miner', mem)]
print("miner mentions:", len(idx))
# Show last few mentions context
for i in idx[-6:]:
    print('---')
    print(mem[max(0,i-400):i+400].replace('\n',' | ')[:800])