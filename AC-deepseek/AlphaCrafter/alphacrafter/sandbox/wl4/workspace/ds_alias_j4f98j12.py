with open('memory.txt') as f:
    lines = f.readlines()
for i in range(48, len(lines)):
    print(lines[i])
