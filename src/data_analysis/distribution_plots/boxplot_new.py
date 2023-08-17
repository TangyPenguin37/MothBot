import os
from enum import IntEnum
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

class Location(IntEnum):
    FRONT = 0
    REAR = 1

class Side(IntEnum):
    VENTRAL = 0
    DORSAL = 1

class Wing(IntEnum):
    LEFT = 0
    RIGHT = 1

LOCATION = Location.REAR
SIDE = Side.DORSAL
WING = Wing.RIGHT

csvData = np.loadtxt('data/split_data/train/with_hybrids/new_results.csv',
                     delimiter=',',
                     skiprows=1,
                     usecols=range(1, 45))

labels = np.loadtxt('data/split_data/train/with_hybrids/new_results.csv',
                    delimiter=',',
                    dtype=np.str0)

labels = labels[0, 1:]

print(labels)

print(csvData.shape)

# filter data by wing
filtered = csvData[csvData[:, 0] == 1]
filtered = filtered[filtered[:, 2] == 0]
# filtered = filtered[filtered[:, 2] == LOCATION]

# print(filtered)

# filtered = csvData

print(filtered.shape)

for j in trange(3, 42):
    data = [(filtered[i, j], filtered[i, -1])
            for i in range(filtered.shape[0])]

    data = np.array(data)

    # print(data)

    # split into three columns by class
    data = [
        data[data[:, -1] == 0][:, 0], data[data[:, -1] == 1][:, 0],
        data[data[:, -1] == 2][:, 0]
    ]

    # print(data)

    plt.boxplot(data, labels=['arm', 'hyb', 'zea'])

    plt.title(labels[j])

    # plt.show()

    if not os.path.exists('plots'):
        os.makedirs('plots')

    plt.savefig('plots/' + labels[j].lower() + '.png')

    plt.clf()