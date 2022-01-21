# %%
import numpy as np
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import csv
import os
import pandas as pd

# read into dataframes with headers
import_path = os.getcwd() + '\\ana1\\analysis.csv'
model_data = pd.read_csv(import_path, header=None, skiprows=[0])
headers = model_data.T.iloc[0]
model_data = pd.DataFrame(model_data.T.values[1:], columns=headers)

import_path = os.getcwd() + '\\ana1\\human_data.csv'
human_data = pd.read_csv(import_path, header=None)
headers = human_data.T.iloc[0]
human_data = pd.DataFrame(human_data.T.values[1:], columns=headers)

# convert it to one dataframe
# first dimension
nbacks = ['0-back', '1-back', '2-back', '3-back', '4-back']

dfs = [nbacks, list(model_data['avg_SRR_high']), list(model_data['avg_SRR_con']),
       list(human_data['Highway_SRR']), list(human_data['Construction_SRR'])]
df = pd.DataFrame.from_records(dfs[1:], columns=dfs[0])

# second dimension
labels = ['Model_Highway', 'Model_Construction',
          'Human_Highway', 'Human_Construction']
data = pd.DataFrame(columns=labels)
for i in range(len(labels)):
    data[labels[i]] = df.T[i]
# fsd data = data

# %% plotting results

labels = ['0-back', '1-back', '2-back', '3-back', '4-back']
x = np.arange(len(labels))
width = 0.3

csfont = {'family': 'Times new roman',
          'size': 14.0}
plt.rc('font', **csfont)
fig, ax1 = plt.subplots()

highway_color = 'tab:orange'
construction_color = 'tab:blue'

# first axis
rects1 = ax1.bar(x - width/2, data['Model_Highway'], width,alpha=0.8,
                 label='Highway (Model)', ecolor='grey', edgecolor='black', color=highway_color)
rects2 = ax1.bar(x + width/2, data['Model_Construction'], width, alpha=0.8,
                 label='Construction (Model)', ecolor='grey', edgecolor='black', color=construction_color)

ax1.set_ylabel('Steering reversal rates in Hz (Model)')
ax1.set_xlabel('N-back levels')
ax1.set_ylim(0.265, 0.31)
#ax.set_title('Mean SRRs')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)

# second axis
ax2 = ax1.twinx()
scat1 = ax2.scatter(x - width/2, data['Human_Highway'], s=40.0, 
                    marker='o', label='Highway (Humans)', color='coral', edgecolor='black', linewidths=1)
scat2 = ax2.scatter(x + width/2, data['Human_Construction'], s=40.0, 
                    marker='o', label='Construction (Humans)', color='tab:cyan', edgecolor='black', linewidths=1)
ax2.set_ylabel('Steering reversal rates in Hz (Humans)')
ax2.set_ylim(0.0, 0.03)

# legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2,
           loc='upper right', prop={'size': 10})

fig.tight_layout()
plt.savefig('ana1/steering_reversals.svg')
plt.show()

# %% import error rates

# read into dataframes with headers
import_path = os.getcwd() + '\\ana1\\analysis.csv'
model_data = pd.read_csv(import_path, header=None, skiprows=[0])
headers = model_data.T.iloc[0]
model_data = pd.DataFrame(model_data.T.values[1:], columns=headers)

import_path = os.getcwd() + '\\ana1\\human_data.csv'
human_data = pd.read_csv(import_path, header=None)
headers = human_data.T.iloc[0]
human_data = pd.DataFrame(human_data.T.values[1:], columns=headers)

# convert it to one dataframe
# first dimension
nbacks = ['0-back', '1-back', '2-back', '3-back', '4-back']

dfs = [nbacks, list(model_data['avg_error_rate_high']), list(model_data['avg_error_rate_con']),
       list(human_data['Highway_error']), list(human_data['Construction_error'])]
df = pd.DataFrame.from_records(dfs[1:], columns=dfs[0])

# second dimension
labels = ['Model_Highway', 'Model_Construction',
          'Human_Highway', 'Human_Construction']
data = pd.DataFrame(columns=labels)
for i in range(len(labels)):
    data[labels[i]] = df.T[i]

# %% error rates

# plotting results
labels = ['0-back', '1-back', '2-back', '3-back', '4-back']
x = np.arange(len(labels))
width = 0.3
highway_color = 'tab:orange'
construction_color = 'tab:blue'

# error rates
fig, ax1 = plt.subplots()
rects1 = ax1.bar(x - width/2, data['Model_Highway'], width, alpha=0.8,
                 label='Highway (Model)', edgecolor='black', color=highway_color)  # , yerr=sem_error_high, capsize=5, ecolor='grey')
rects2 = ax1.bar(x + width/2, data['Model_Construction'], width,  alpha=0.8,
                 label='Construction (Model)', edgecolor='black', color=construction_color)  # , yerr=sem_error_con, capsize=5, ecolor='grey')

ax1.set_ylabel('Error rates in %')
ax1.set_ylim(0.0, 0.25)
ax1.set_xticks(x)
ax1.set_xticklabels(labels)

# second axis
ax2 = ax1.twinx()
scat1 = ax2.scatter(x - width/2, data['Human_Highway'], s=40.0,
                    marker='o', label='Highway (Humans)', color='coral', edgecolor='black', linewidths=1)
scat2 = ax2.scatter(x + width/2, data['Human_Construction'], s=40.0,
                    marker='o', label='Construction (Humans)', color='tab:cyan', edgecolor='black', linewidths=1)
ax2.set_ylim(0.0, 0.3)
ax2.set_yticklabels([])
ax2.set_yticks([])

# legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
# , bbox_to_anchor=(1.05, 1.00))
ax2.legend(lines + lines2, labels + labels2, loc=2, prop={'size': 10})

fig.tight_layout()
plt.savefig('ana1/error_rates.svg')
plt.show()

# %%
def format_str(unformatted_str):
    """
    Formats the output into a multi column list,  output1 | output2 | output3
    """
    output = []
    for word in enumerate(unformatted_str):
        if word != '':
            output.append(word[1])
    return output
# %% driving path of example participant
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import glob
import pandas as pd

import_path = os.getcwd() + '\data1\*behavior*.txt'
output_path = os.getcwd()
path_list = list(enumerate(glob.glob(import_path)))
if (path_list == []):
    raise Exception('There is no data in the specified folder!')
srate = 0.05
nbacklevels = ['0back', '1back', '2back', '3back', '4back']

i = rd.randrange(len(path_list))
filename = path_list[i]

f = open(filename[1], 'r')
lines = [line.rstrip('\n') for line in f]

data_subj = []
for row, word in enumerate(lines):
    unformatted_str = word.split('|')
    data_subj.append(format_str(unformatted_str))

data = pd.DataFrame(data_subj[1:], columns=data_subj[0])

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')

def init():
    ax.set_xlim(0,5)
    ax.set_ylim(0,5)
    return ln,

def update(frame):
    xdata.append(frame)
    ydata.append(frame)
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=data['lanepos'],
                    init_func=init, blit=True)
plt.show()
ani.save('test.gif', writer='pillow')

# %%
