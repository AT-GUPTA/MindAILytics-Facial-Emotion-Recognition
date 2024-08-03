import os
from pathlib import Path


dir_path = os.path.dirname(os.path.realpath(__file__))

i = 0

happy = 0
neutral = 0
surprised = 0
focused = 0

datasets = os.path.abspath(os.path.join(dir_path, os.pardir)) + '\\Datasets'

#Loop through all the Subfolders of each Dataset and prints the number of pictures included in the subfolders
#Then Prints total number of pictures for each class

for dirname, dirnames, filenames in os.walk(datasets):
    if '.git' in dirnames:
        dirnames.remove('.git')
    for subdirname in dirnames:
        p = Path(os.path.join(dirname, subdirname))
        if "happy" in subdirname:
            print(f'{str(p.parent)}\\{p.stem}: {len(os.listdir(os.path.join(dirname, subdirname)))}')
            happy += len(os.listdir(os.path.join(dirname, subdirname)))
        if "surprised" in subdirname:
            print(f'{str(p.parent)}\\{p.stem}: {len(os.listdir(os.path.join(dirname, subdirname)))}')
            surprised += len(os.listdir(os.path.join(dirname, subdirname)))
        if "neutral" in subdirname:
            print(f'{str(p.parent)}\\{p.stem}: {len(os.listdir(os.path.join(dirname, subdirname)))}')
            neutral += len(os.listdir(os.path.join(dirname, subdirname)))
        if "focused" in subdirname:
            print(f'{str(p.parent)}\\{p.stem}: {len(os.listdir(os.path.join(dirname, subdirname)))}')
            focused += len(os.listdir(os.path.join(dirname, subdirname)))

print(f'{happy=}, {focused=}, {surprised=}, {neutral=}')
