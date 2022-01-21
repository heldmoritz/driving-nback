# %% import
import glob
import os
from numpy.core.numeric import NaN
import pandas as pd
import numpy as np

# functions


def format_str(unformatted_str):
    """
    Formats the trace string into a three column list,  times | module | action
    """
    output = []
    for ind, word in enumerate(unformatted_str):
        if word != '':
            output.append(word)
    return output


# import the data
import_path = os.getcwd() + '/data/*.txt'
output_path = os.getcwd()
trial_list = glob.glob(import_path)
subj = list(enumerate(trial_list))

for s, filename in enumerate(subj):
    # example for one trial
    f = open(trial_list[s], 'r')
    lines = f.readlines()

    data = []
    for row, word in enumerate(lines):
        unformatted_str = word.split('  ')
        data.append(format_str(unformatted_str))

    df = pd.DataFrame(data, columns=['Time', 'Module', 'Action'])
    for i, word in enumerate(df.columns):
        df[word] = df[word].str.strip()

    # Determine onsets and offsets for every module

    # pre-allocating the variables
    visual_location = []
    vision = []
    procedural = []
    imaginal = []
    declarative = []
    chunk_i = []

    # looping through the trace
    for i, row in df.iterrows():
        # visual-location
        if 'find-location' in row['Action']:
            visual_location.append(row['Time'])
            visual_location.append(float(row['Time']) + 0.001)

        # vision
        if 'vision' in row['Module'] and \
                'find-location' not in row['Action'] and \
                'error' not in row['Action'] and \
                'unrequested' not in row['Action']:
            vision.append(df['Time'][i-1])
            vision.append(row['Time'])

        # procedural
        if 'procedural' in row['Module'] and \
                'start' not in row['Action']:
            procedural.append(float(row['Time']) - 0.050)
            procedural.append(row['Time'])

        # imaginal
        if 'imaginal' in row['Module'] and \
                'init' not in row['Action']:
            imaginal.append(float(row['Time']) - 0.200)
            imaginal.append(row['Time'])

        # declarative
        if 'declarative' in row['Module']:
            if 'store chunk' in row['Action']:
                declarative.append(row['Time'])
                declarative.append(float(row['Time']) + 0.001)
            if 'start-retrieval' in row['Action']:
                declarative.append(str(row['Time']))
                chunk_i = i
                declarative.append('placeholder' + str(chunk_i))
                # If the chunk is not retrieved, it will not accidently receive an offset
            if ('retrieved-chunk') in row['Action']:
                declarative = [str(decl).replace(
                    ('placeholder' + str(chunk_i)), str(row['Time'])) for decl in declarative]
                # The latest chunk which a retrieval was issued for will receive an offset if it is retrieved
                # XXX code review?

    # resolve chunks that weren't retrieved
    for i in range(len(declarative)):
        if 'placeholder' in str(declarative[i]):
            declarative[i] = NaN

    # write neatly to a file
    modules = [visual_location, vision, procedural, imaginal, declarative]
    modules_str = ['visual_location', 'vision',
                   'procedural', 'imaginal', 'declarative']
    output = pd.DataFrame(columns=['Module', 'Onsets', 'Offsets'])

    for i in range(len(modules)):
        current_mod = modules[i]
        current_mod_str = modules_str[i]
        count = 0
        data = np.ndarray([int(len(current_mod)/2), 2])
        for j in range(len(current_mod)):
            if j % 2 == 0:
                data[count, 0] = current_mod[j]
            else:
                data[count, 1] = current_mod[j]
                count += 1
        modules[i] = data

    with open('./onsets/onoff_subj' + str(s) + '.txt', 'w') as f:
        f.write('Onset' + ' Offset' + '\n')
        for i in range(len(modules)):
            np.savetxt(f, modules[i], delimiter=' ',
                       fmt="%.3f", header=modules_str[i])

# %% read it back from files

import_path = os.getcwd() + '/trace/*.txt'
filenames = glob.glob(import_path)

for s, filename in enumerate(filenames):
    x = 1