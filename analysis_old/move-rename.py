# %%
import glob
import os
import shutil
from pathlib import Path
import re

path_origin = os.getcwd() + '\data\\'
path_import = path_origin + '*behavior*.txt'
path_destination = str(Path(path_import).parent.parent) + '\data1\\'
list_origin = list(enumerate(glob.glob(path_import)))
nbacklevels = ['0back', '1back', '2back', '3back', '4back']

for i, file in list_origin:
    
    file = file.replace(path_origin, '')
    path_file = path_destination + file
    
    list_destination = list(enumerate(glob.glob(path_destination + '*behavior*')))
    
    # check if file exists in destination folder
    if (any((path_file) in s for s in list_destination)):        
        
        # find n-back
        nback = [poss for poss in nbacklevels if (poss in path_file)][0]
        
        # find last existing participant for nback level and increment number
        ind = max(loc for loc, val in enumerate(list_destination) if nback in val[1])
        max_participant = list_destination[ind][1]
        max_participant = max_participant.split('behavior_', 1)[1].replace('.txt', '')
        ind = "{:02d}".format(int(max_participant) + 1)
        path_file = re.sub('behavior_(?:.{2})', 'behavior_' + ind, path_file)

    # move file
    shutil.move(path_origin + file, path_file)
# %%
