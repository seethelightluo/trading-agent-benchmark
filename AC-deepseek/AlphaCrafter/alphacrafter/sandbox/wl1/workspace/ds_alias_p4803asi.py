with open('memory.txt') as f:
    content = f.read()
lines = content.split('\n')
# Last non-empty line is the 20350223 cycle entry
print(lines[-2][:6000])