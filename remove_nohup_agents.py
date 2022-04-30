import os
from pprint import pprint

# ui_to_confirm = True
ui_to_confirm = False
# --------------------

files = os.listdir()
agent_files = []
for file in files:
    if 'nohup' in file and '.out' in file:
        agent_files.append(file)

print(f"found {len(agent_files)} file names containing 'nohup' and '.out':")
pprint(agent_files)

if ui_to_confirm==False or input(f"delete {len(agent_files)} files found? y/[n] ")=='y':
    for file in agent_files:
        os.remove(file)
    print("files removed")