from enum import IntEnum
import colorsys
import numpy as np
import matplotlib.pyplot as plt

class Color(IntEnum):
    COLOR0 = 0
    COLOR1 = 1
    COLOR2 = 2
    COLOR3 = 3

class Location(IntEnum):
    FRONT = 0
    REAR = 1

class Side(IntEnum):
    LEFT = 0
    RIGHT = 1

USE_HSV = False
COLORS = [Color.COLOR1]
LOCATIONS = [Location.FRONT]
SIDES = [Side.RIGHT, Side.LEFT]

filepath = 'data/original_labelled/data_normalized_onehot_simplified.csv'

csvData = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=range(1, 52))

labels = np.loadtxt(filepath,
                    delimiter=',',
                    dtype=np.str0,
                    max_rows=1,
                    usecols=range(1, 52))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for color in COLORS:
    for location in LOCATIONS:
        for side in SIDES:
            columns = [
                x for x in range(2, 51) if "colour_" + str(color) in labels[x]
            ]
            columns = sorted([0, 1, 50] + columns)

            data = csvData[csvData[:, 0] == location]
            data = data[data[:, 1] == side]

            data = data[:, columns]

            # plot on 3d scatter plot

            scatterColours = ['r' if x == 1 else 'b' for x in data[:, -1]]

            if USE_HSV:
                for i in range(len(data)):
                    rgb = data[i, 2:5]
                    hsv = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
                    data[i, 2:5] = hsv

                hsv = data[:, 2:5]
                print(hsv)

                ax.scatter(hsv[:, 0],
                           hsv[:, 1],
                           hsv[:, 2],
                           c=scatterColours,
                           marker='o')
            else:
                ax.scatter(data[:, 2],
                           data[:, 3],
                           data[:, 4],
                           c=scatterColours,
                           marker='o')

            print("sample size: " + str(len(data)))

ax.set_xlabel("red")
ax.set_ylabel("green")
ax.set_zlabel("blue")

plt.show()